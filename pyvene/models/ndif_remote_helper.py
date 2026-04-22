"""
Pure nnsight helper for remote execution.
This module intentionally does NOT import pyvene to avoid whitelist issues.

CRITICAL: This file must remain import-free (only torch and basic Python built-ins allowed).
All intervention logic must be expressed as pure tensor operations.
"""

import torch

CONST_INPUT_HOOK = "input"
CONST_OUTPUT_HOOK = "output"


def _get_module_output(module_hook, hook_type):
    """Return the proxy output for a module hook inside a trace context."""
    if hook_type == CONST_INPUT_HOOK:
        return module_hook.input
    return module_hook.output


def _get_act(output):
    """Extract the activation tensor from a possibly-tuple proxy."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _set_act(output, value):
    """In-place assign value into the activation (handles tuple outputs)."""
    if isinstance(output, tuple):
        output[0][:] = value
    else:
        output[:] = value


def execute_remote_intervention(
    model,
    base,
    sources,
    intervention_specs,
    intervention_group,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """
    Execute interventions on NDIF remote backend using pure nnsight operations.

    Supports all pyvene intervention types:
      - CollectIntervention
      - VanillaIntervention
      - ZeroIntervention
      - AdditionIntervention
      - SubtractionIntervention
      - NoiseIntervention
      - LambdaIntervention
      - All TrainableInterventions (via get_remote_weights())
    """
    if output_module is None:
        output_module = model.lm_head  # fallback for models without _get_output_module

    # Categorize specs
    collect_specs     = [s for s in intervention_specs if s.get('is_collect')]
    vanilla_specs     = [s for s in intervention_specs if s.get('is_vanilla') and not s.get('is_trainable')]
    trainable_specs   = [s for s in intervention_specs if s.get('is_trainable') and not s.get('is_collect')]
    zero_specs        = [s for s in intervention_specs if s.get('is_zero')]
    addition_specs    = [s for s in intervention_specs if s.get('is_addition')]
    subtraction_specs = [s for s in intervention_specs if s.get('is_subtraction')]
    noise_specs       = [s for s in intervention_specs if s.get('is_noise')]
    lambda_specs      = [s for s in intervention_specs if s.get('is_lambda')]

    # Source-constant: no source trace needed (zero, noise, constant-source vanilla/addition)
    sourceless_keys = set(
        s['key'] for s in zero_specs + noise_specs
        if s.get('is_source_constant')
    )
    # Also mark vanilla/addition/subtraction specs that have a pre-set source_representation
    for s in vanilla_specs + addition_specs + subtraction_specs + lambda_specs:
        if s.get('source_representation') is not None:
            sourceless_keys.add(s['key'])

    # Specs that need a source activation collected from sources[group_id]
    needs_source = (
        vanilla_specs + addition_specs + subtraction_specs + trainable_specs +
        [s for s in lambda_specs if not s.get('is_source_constant')]
    )
    needs_source = [s for s in needs_source if s['key'] not in sourceless_keys]

    # -----------------------------------------------------------------------
    # COLLECT-ONLY path (no modification)
    # -----------------------------------------------------------------------
    if collect_specs and not vanilla_specs and not trainable_specs \
            and not zero_specs and not addition_specs and not subtraction_specs \
            and not noise_specs and not lambda_specs:
        collected_activations = {}
        model_out = None

        for spec in collect_specs:
            with model.trace(base, remote=True, **kwargs):
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                saved = _get_act(output).save()
                model_out = output_module.output.save()
            collected_activations[spec['key']] = saved

        return {'output': model_out, 'activations': collected_activations}

    # -----------------------------------------------------------------------
    # GENERAL path: collect sources then apply to base
    # -----------------------------------------------------------------------
    source_activations = {}
    collected_activations = {}

    # Pre-compute group → specs mapping for source collection
    specs_by_group = {}
    for spec in needs_source:
        gid = spec['group_id']
        specs_by_group.setdefault(gid, []).append(spec)

    with model.session(remote=True):
        # --- Source collection pass ---
        for group_id, specs_in_group in specs_by_group.items():
            if sources is None or group_id >= len(sources) or sources[group_id] is None:
                continue
            with model.trace(sources[group_id]):
                for spec in specs_in_group:
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[spec['key']] = _get_act(output).save()

        # --- Base intervention pass ---
        with model.trace(base, **kwargs):

            # Collect interventions (no modification)
            for spec in collect_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                collected_activations[spec['key']] = act.save()

            # Zero interventions
            for spec in zero_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, torch.zeros_like(act))

            # Noise interventions
            for spec in noise_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                interchange_d = spec.get('interchange_dim')
                noise_level = spec.get('noise_level', 0.0)
                if interchange_d is not None:
                    noisy = act.clone()
                    noisy[..., :interchange_d] = (
                        act[..., :interchange_d]
                        + torch.randn_like(act[..., :interchange_d]) * noise_level
                    )
                    _set_act(output, noisy)
                else:
                    _set_act(output, act + torch.randn_like(act) * noise_level)

            # Vanilla interventions (simple swap)
            for spec in vanilla_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                src = src.to(dtype=_get_act(
                    _get_module_output(spec['module_hook'], spec['hook_type'])
                ).dtype)
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                _set_act(output, src)

            # Addition interventions
            for spec in addition_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, act + src.to(act.device, act.dtype))

            # Subtraction interventions
            for spec in subtraction_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, act - src.to(act.device, act.dtype))

            # Lambda interventions
            for spec in lambda_specs:
                fn = spec.get('lambda_fn')
                if fn is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                result = fn(act, src)
                _set_act(output, result)

            # Trainable interventions (rotation-based)
            for spec in trainable_specs:
                weights = spec.get('intervention_weights')
                if not weights:
                    # Custom trainable without get_remote_weights: fall back to vanilla swap
                    src = source_activations.get(spec['key'])
                    if src is not None:
                        output = _get_module_output(spec['module_hook'], spec['hook_type'])
                        _set_act(output, src)
                    continue

                if spec['key'] not in source_activations:
                    continue

                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                base_act = _get_act(output)
                source_act = source_activations[spec['key']]
                subspaces = spec.get('subspaces')
                intervention_type = weights.get('intervention_type', 'rotated_space')

                if intervention_type == 'sigmoid_mask':
                    mask = weights['mask'].to(base_act.device)
                    temperature = weights['temperature']
                    mask_sigmoid = torch.sigmoid(mask / temperature)
                    intervened = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act

                elif intervention_type == 'pca_rotated_space':
                    pca_c = weights['pca_components'].to(base_act.device)
                    pca_m = weights['pca_mean'].to(base_act.device)
                    pca_s = weights['pca_std'].to(base_act.device)
                    base_norm = (base_act.to(pca_c.dtype) - pca_m) / pca_s
                    src_norm = (source_act.to(pca_c.dtype) - pca_m) / pca_s
                    rot_base = torch.matmul(base_norm, pca_c.T)
                    rot_src = torch.matmul(src_norm, pca_c.T)
                    interchange_d = weights.get('interchange_dim')
                    if interchange_d is not None:
                        rot_base[..., :interchange_d] = rot_src[..., :interchange_d]
                    else:
                        rot_base = rot_src
                    intervened = (torch.matmul(rot_base, pca_c) * pca_s + pca_m).to(base_act.dtype)

                elif intervention_type == 'autoencoder':
                    enc_w = weights['encoder_weight'].to(base_act.device)
                    enc_b = weights['encoder_bias'].to(base_act.device)
                    dec_w = weights['decoder_weight'].to(base_act.device)
                    dec_b = weights['decoder_bias'].to(base_act.device)
                    interchange_d = weights.get('interchange_dim')
                    base_lat = torch.relu(base_act.to(enc_w.dtype) @ enc_w.T + enc_b)
                    src_lat = torch.relu(source_act.to(enc_w.dtype) @ enc_w.T + enc_b)
                    if interchange_d is not None:
                        base_lat[..., :interchange_d] = src_lat[..., :interchange_d]
                    else:
                        base_lat = src_lat
                    intervened = (base_lat @ dec_w.T + dec_b).to(base_act.dtype)

                elif intervention_type == 'jumprelu_autoencoder':
                    W_enc = weights['W_enc'].to(base_act.device)
                    W_dec = weights['W_dec'].to(base_act.device)
                    threshold = weights['threshold'].to(base_act.device)
                    b_enc = weights['b_enc'].to(base_act.device)
                    b_dec = weights['b_dec'].to(base_act.device)
                    interchange_d = weights.get('interchange_dim')
                    pre_base = base_act @ W_enc + b_enc
                    base_lat = (pre_base > threshold) * torch.relu(pre_base)
                    pre_src = source_act @ W_enc + b_enc
                    src_lat = (pre_src > threshold) * torch.relu(pre_src)
                    if interchange_d is not None:
                        base_lat[..., :interchange_d] = src_lat[..., :interchange_d]
                    else:
                        base_lat = src_lat
                    intervened = (base_lat @ W_dec + b_dec).to(base_act.dtype)

                else:
                    # All rotation-based interventions (rotated_space, low_rank_rotated_space,
                    # boundless_rotated_space, sigmoid_mask_rotated_space)
                    rotation_matrix = weights['rotate_layer_weight'].to(base_act.device)
                    rotated_base = torch.matmul(base_act.to(rotation_matrix.dtype), rotation_matrix)
                    rotated_source = torch.matmul(source_act.to(rotation_matrix.dtype), rotation_matrix)
                    diff = rotated_source - rotated_base

                    if intervention_type == 'boundless_rotated_space':
                        intervention_boundaries = weights['intervention_boundaries'].to(base_act.device)
                        temperature = weights['temperature']
                        intervention_population = weights['intervention_population'].to(base_act.device)
                        embed_dim = weights['embed_dim']
                        batch_size = base_act.shape[0]
                        intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
                        positions = intervention_population.repeat(batch_size, 1)
                        boundary_val = intervention_boundaries[0] * embed_dim
                        boundary_mask = torch.sigmoid(temperature * (boundary_val - positions))
                        boundary_mask = boundary_mask.to(rotated_base.dtype)
                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

                    elif intervention_type == 'sigmoid_mask_rotated_space':
                        masks = weights['masks'].to(base_act.device)
                        temperature = weights['temperature']
                        batch_size = base_act.shape[0]
                        boundary_mask = torch.sigmoid(masks / temperature)
                        boundary_mask = (
                            torch.ones(batch_size, device=base_act.device).unsqueeze(-1) * boundary_mask
                        ).to(rotated_base.dtype)
                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

                    elif subspaces is not None:
                        subspace_partition = weights.get('subspace_partition')
                        use_fast = weights.get('use_fast', False)
                        can_use_fast = use_fast or (len(set(tuple(s) for s in subspaces)) == 1)
                        if can_use_fast:
                            sel = subspaces[0] if subspace_partition is None else [
                                i for sub in subspaces[0] for i in subspace_partition[sub]
                            ]
                            batched_subspace = diff[..., sel].unsqueeze(1)
                            batched_weights = rotation_matrix[..., sel].T
                            intervened = (base_act + torch.matmul(batched_subspace, batched_weights).squeeze(1)).to(base_act.dtype)
                        else:
                            batched_subspace, batched_weights_list = [], []
                            for example_i in range(len(subspaces)):
                                sel = [i for sub in subspaces[example_i] for i in subspace_partition[sub]]
                                batched_subspace.append(diff[example_i, sel].unsqueeze(0))
                                batched_weights_list.append(rotation_matrix[..., sel].T)
                            batched_subspace = torch.stack(batched_subspace, dim=0)
                            bw = torch.stack(batched_weights_list, dim=0)
                            intervened = (base_act + torch.matmul(batched_subspace, bw).squeeze(1)).to(base_act.dtype)

                    else:
                        intervened = (base_act + torch.matmul(diff, rotation_matrix.T)).to(base_act.dtype)

                if isinstance(output, tuple):
                    output[0][:] = intervened
                else:
                    output[:] = intervened

            counterfactual_outputs = output_module.output.save()

    return {
        'output': counterfactual_outputs,
        'activations': collected_activations,
    }


def execute_remote_generate(
    model,
    base,
    sources,
    intervention_specs,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """
    Execute generation with interventions on NDIF remote backend.

    Uses nnsight's model.generate() context. Supports all intervention types.
    Returns the full generation output (token ids) via model.output.save().
    """
    if output_module is None:
        output_module = model.lm_head

    # Categorize specs (same taxonomy as execute_remote_intervention)
    collect_specs     = [s for s in intervention_specs if s.get('is_collect')]
    vanilla_specs     = [s for s in intervention_specs if s.get('is_vanilla') and not s.get('is_trainable')]
    trainable_specs   = [s for s in intervention_specs if s.get('is_trainable') and not s.get('is_collect')]
    zero_specs        = [s for s in intervention_specs if s.get('is_zero')]
    addition_specs    = [s for s in intervention_specs if s.get('is_addition')]
    subtraction_specs = [s for s in intervention_specs if s.get('is_subtraction')]
    noise_specs       = [s for s in intervention_specs if s.get('is_noise')]
    lambda_specs      = [s for s in intervention_specs if s.get('is_lambda')]

    sourceless_keys = set(
        s['key'] for s in zero_specs + noise_specs if s.get('is_source_constant')
    )
    for s in vanilla_specs + addition_specs + subtraction_specs + lambda_specs:
        if s.get('source_representation') is not None:
            sourceless_keys.add(s['key'])

    needs_source = (
        vanilla_specs + addition_specs + subtraction_specs + trainable_specs +
        [s for s in lambda_specs if not s.get('is_source_constant')]
    )
    needs_source = [s for s in needs_source if s['key'] not in sourceless_keys]

    specs_by_group = {}
    for spec in needs_source:
        specs_by_group.setdefault(spec['group_id'], []).append(spec)

    source_activations = {}
    collected_activations = {}

    with model.session(remote=True):
        # Source collection (use trace, not generate)
        for group_id, specs_in_group in specs_by_group.items():
            if sources is None or group_id >= len(sources) or sources[group_id] is None:
                continue
            with model.trace(sources[group_id]):
                for spec in specs_in_group:
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[spec['key']] = _get_act(output).save()

        # Generation with interventions applied at every forward step
        with model.generate(base, **kwargs):
            for spec in collect_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                collected_activations[spec['key']] = _get_act(output).save()

            for spec in zero_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                _set_act(output, torch.zeros_like(_get_act(output)))

            for spec in noise_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                noise_level = spec.get('noise_level', 0.0)
                interchange_d = spec.get('interchange_dim')
                if interchange_d is not None:
                    noisy = act.clone()
                    noisy[..., :interchange_d] += torch.randn_like(act[..., :interchange_d]) * noise_level
                    _set_act(output, noisy)
                else:
                    _set_act(output, act + torch.randn_like(act) * noise_level)

            for spec in vanilla_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, src.to(act.device, act.dtype))

            for spec in addition_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, act + src.to(act.device, act.dtype))

            for spec in subtraction_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, act - src.to(act.device, act.dtype))

            for spec in lambda_specs:
                fn = spec.get('lambda_fn')
                if fn is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                _set_act(output, fn(act, src))

            for spec in trainable_specs:
                weights = spec.get('intervention_weights')
                if not weights:
                    src = source_activations.get(spec['key'])
                    if src is not None:
                        output = _get_module_output(spec['module_hook'], spec['hook_type'])
                        _set_act(output, src)
                    continue
                if spec['key'] not in source_activations:
                    continue

                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                base_act = _get_act(output)
                source_act = source_activations[spec['key']]
                intervention_type = weights.get('intervention_type', 'rotated_space')

                if intervention_type == 'sigmoid_mask':
                    mask = weights['mask'].to(base_act.device)
                    temperature = weights['temperature']
                    mask_sigmoid = torch.sigmoid(mask / temperature)
                    intervened = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act
                else:
                    rotation_matrix = weights['rotate_layer_weight'].to(base_act.device)
                    rotated_base = torch.matmul(base_act.to(rotation_matrix.dtype), rotation_matrix)
                    rotated_source = torch.matmul(source_act.to(rotation_matrix.dtype), rotation_matrix)
                    diff = rotated_source - rotated_base

                    if intervention_type == 'boundless_rotated_space':
                        intervention_boundaries = weights['intervention_boundaries'].to(base_act.device)
                        temperature = weights['temperature']
                        intervention_population = weights['intervention_population'].to(base_act.device)
                        embed_dim = weights['embed_dim']
                        batch_size = base_act.shape[0]
                        intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
                        positions = intervention_population.repeat(batch_size, 1)
                        boundary_val = intervention_boundaries[0] * embed_dim
                        boundary_mask = torch.sigmoid(temperature * (boundary_val - positions)).to(rotated_base.dtype)
                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)
                    elif intervention_type == 'sigmoid_mask_rotated_space':
                        masks = weights['masks'].to(base_act.device)
                        temperature = weights['temperature']
                        batch_size = base_act.shape[0]
                        boundary_mask = torch.sigmoid(masks / temperature)
                        boundary_mask = (
                            torch.ones(batch_size, device=base_act.device).unsqueeze(-1) * boundary_mask
                        ).to(rotated_base.dtype)
                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)
                    else:
                        intervened = (base_act + torch.matmul(diff, rotation_matrix.T)).to(base_act.dtype)

                if isinstance(output, tuple):
                    output[0][:] = intervened
                else:
                    output[:] = intervened

            gen_output = model.generator.output.save()

    return {'output': gen_output, 'activations': collected_activations}


def execute_remote_serial_intervention(
    model,
    base,
    sources,
    intervention_specs,
    intervention_group,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """
    Execute serial interventions on NDIF remote backend.

    Serial mode: each group's source is traced (with prior group activations applied),
    then the final base trace applies all accumulated activations.
    """
    if output_module is None:
        output_module = model.lm_head

    sorted_group_ids = sorted(intervention_group.keys())
    source_activations = {}

    with model.session(remote=True):
        # For each group in order: collect its activation (applying prior interventions)
        for group_id in sorted_group_ids:
            keys = intervention_group[group_id]
            source = sources[group_id] if sources and group_id < len(sources) else None
            if source is None:
                continue

            with model.trace(source):
                # Apply interventions from all prior groups
                for prior_id in sorted_group_ids:
                    if prior_id >= group_id:
                        break
                    for spec in intervention_specs:
                        if spec['group_id'] == prior_id and spec['key'] in source_activations:
                            output = _get_module_output(spec['module_hook'], spec['hook_type'])
                            src = source_activations[spec['key']]
                            if spec.get('is_vanilla'):
                                _set_act(output, src)
                            elif spec.get('is_addition'):
                                act = _get_act(output)
                                _set_act(output, act + src.to(act.device, act.dtype))

                # Collect current group's activations
                for key in keys:
                    spec = next(s for s in intervention_specs if s['key'] == key)
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[key] = _get_act(output).save()

        # Final pass: apply all collected activations to base
        with model.trace(base, **kwargs):
            for spec in intervention_specs:
                if spec['key'] not in source_activations:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                src = source_activations[spec['key']]
                act = _get_act(output)

                if spec.get('is_vanilla'):
                    _set_act(output, src.to(act.device, act.dtype))
                elif spec.get('is_addition'):
                    _set_act(output, act + src.to(act.device, act.dtype))
                elif spec.get('is_subtraction'):
                    _set_act(output, act - src.to(act.device, act.dtype))
                elif spec.get('is_trainable') and spec.get('intervention_weights'):
                    weights = spec['intervention_weights']
                    intervention_type = weights.get('intervention_type', 'rotated_space')
                    if intervention_type == 'sigmoid_mask':
                        mask = weights['mask'].to(act.device)
                        temperature = weights['temperature']
                        mask_sigmoid = torch.sigmoid(mask / temperature)
                        _set_act(output, (1.0 - mask_sigmoid) * act + mask_sigmoid * src)
                    else:
                        rotation_matrix = weights['rotate_layer_weight'].to(act.device)
                        rotated_base = torch.matmul(act.to(rotation_matrix.dtype), rotation_matrix)
                        rotated_source = torch.matmul(src.to(rotation_matrix.dtype), rotation_matrix)
                        diff = rotated_source - rotated_base
                        intervened = (act + torch.matmul(diff, rotation_matrix.T)).to(act.dtype)
                        _set_act(output, intervened)
                else:
                    _set_act(output, src.to(act.device, act.dtype))

            counterfactual_outputs = output_module.output.save()

    return {'output': counterfactual_outputs, 'activations': {}}
