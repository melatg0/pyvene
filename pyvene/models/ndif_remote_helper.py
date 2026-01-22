"""
Pure nnsight helper for remote execution.
This module intentionally does NOT import pyvene to avoid whitelist issues.

CRITICAL: This file must remain import-free (only torch and basic Python built-ins allowed).
All intervention logic must be expressed as pure tensor operations.
"""

import torch

CONST_INPUT_HOOK = "input"
CONST_OUTPUT_HOOK = "output"


def apply_rotation_intervention(base_act, source_act, weights, subspaces=None):
    """
    Apply rotation-based intervention using pure tensor operations.

    Mirrors the logic from LowRankRotatedSpaceIntervention.forward() and
    RotatedSpaceIntervention.forward(). No pyvene imports allowed.

    Args:
        base_act: Base activation tensor
        source_act: Source activation tensor
        weights: Dict with rotation weights from get_remote_weights()
        subspaces: Optional list of subspace indices per example

    Returns:
        Intervened activation tensor
    """
    rotation_matrix = weights['rotate_layer_weight']
    intervention_type = weights.get('intervention_type', 'rotated_space')
    subspace_partition = weights.get('subspace_partition')
    use_fast = weights.get('use_fast', False)

    # Project to rotated space
    rotated_base = torch.matmul(base_act.to(rotation_matrix.dtype), rotation_matrix)
    rotated_source = torch.matmul(source_act.to(rotation_matrix.dtype), rotation_matrix)

    diff = rotated_source - rotated_base

    if intervention_type == 'boundless_rotated_space':
        # BoundlessRotatedSpaceIntervention logic
        intervention_boundaries = weights.get('intervention_boundaries')
        temperature = weights.get('temperature')
        intervention_population = weights.get('intervention_population')
        embed_dim = weights.get('embed_dim')
        batch_size = base_act.shape[0]

        # Compute boundary mask using sigmoid boundary
        intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
        # sigmoid_boundary approximation using torch operations
        positions = intervention_population.repeat(batch_size, 1).to(rotation_matrix.device)
        boundary_val = intervention_boundaries[0] * embed_dim
        # Sigmoid boundary: 1 / (1 + exp(-temperature * (boundary - position)))
        boundary_mask = torch.sigmoid(temperature * (boundary_val - positions))
        boundary_mask = boundary_mask.to(rotated_base.dtype)

        # Interpolate between base and source in rotated space
        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
        output = torch.matmul(rotated_output, rotation_matrix.T)
        return output.to(base_act.dtype)

    elif intervention_type == 'sigmoid_mask_rotated_space':
        # SigmoidMaskRotatedSpaceIntervention logic
        masks = weights.get('masks')
        temperature = weights.get('temperature')
        batch_size = base_act.shape[0]

        boundary_mask = torch.sigmoid(masks / temperature)
        boundary_mask = torch.ones(batch_size, device=base_act.device).unsqueeze(dim=-1) * boundary_mask
        boundary_mask = boundary_mask.to(rotated_base.dtype)

        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
        output = torch.matmul(rotated_output, rotation_matrix.T)
        return output.to(base_act.dtype)

    elif intervention_type == 'sigmoid_mask':
        # SigmoidMaskIntervention logic (no rotation, just masking)
        mask = weights.get('mask')
        temperature = weights.get('temperature')
        mask_sigmoid = torch.sigmoid(mask / temperature)
        output = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act
        return output

    # Handle subspaces if specified (for low_rank_rotated_space and rotated_space)
    if subspaces is not None:
        # Check if we can use fast path (all examples have same subspaces)
        can_use_fast = use_fast or (len(set(tuple(s) for s in subspaces)) == 1)

        if can_use_fast:
            # Fast path: all examples have same subspaces
            if subspace_partition is None:
                sel_subspace_indices = subspaces[0]
            else:
                sel_subspace_indices = []
                for subspace in subspaces[0]:
                    sel_subspace_indices.extend(subspace_partition[subspace])

            batched_subspace = diff[..., sel_subspace_indices].unsqueeze(dim=1)
            batched_weights = rotation_matrix[..., sel_subspace_indices].T
            output = base_act + torch.matmul(batched_subspace, batched_weights).squeeze(dim=1)
        else:
            # Per-example subspaces (slower path)
            batched_subspace = []
            batched_weights = []
            for example_i in range(len(subspaces)):
                sel_subspace_indices = []
                for subspace in subspaces[example_i]:
                    sel_subspace_indices.extend(subspace_partition[subspace])
                LHS = diff[example_i, sel_subspace_indices].unsqueeze(dim=0)
                RHS = rotation_matrix[..., sel_subspace_indices].T
                batched_subspace.append(LHS)
                batched_weights.append(RHS)
            batched_subspace = torch.stack(batched_subspace, dim=0)
            batched_weights = torch.stack(batched_weights, dim=0)
            output = base_act + torch.matmul(batched_subspace, batched_weights).squeeze(dim=1)
    else:
        # No subspaces - simple case
        output = base_act + torch.matmul(diff, rotation_matrix.T)

    return output.to(base_act.dtype)


def execute_remote_intervention(
    model,
    base,
    sources,
    intervention_specs,
    intervention_group,
    activations_sources=None,
    **kwargs
):
    """
    Execute interventions on NDIF remote backend using pure nnsight operations.

    This function is intentionally in a separate module that doesn't import pyvene,
    to avoid NDIF whitelist issues when the code is captured and sent to the server.

    IMPORTANT: All code inside trace blocks must be pure nnsight operations.
    No function calls to helper functions, no imports, no complex Python operations.

    Supports:
    - CollectIntervention: Collects activations without modification
    - VanillaIntervention: Simple activation swapping
    - TrainableIntervention: Rotation-based interventions (LowRankRotatedSpaceIntervention, etc.)
    """
    # Identify which specs are collect, vanilla, or trainable
    collect_specs = [s for s in intervention_specs if s.get('is_collect')]
    vanilla_specs = [s for s in intervention_specs if s.get('is_vanilla') and not s.get('is_trainable')]
    trainable_specs = [s for s in intervention_specs if s.get('is_trainable') and not s.get('is_collect')]

    # For collect interventions, collect each one separately
    if collect_specs and not vanilla_specs and not trainable_specs:
        collected_activations = {}

        # Collect each activation in a separate trace
        for spec in collect_specs:
            module_hook = spec['module_hook']
            hook_type = spec['hook_type']

            # Determine what to access BEFORE entering trace
            use_input = (hook_type == "input")

            with model.trace(base, remote=True, **kwargs):
                # INLINE: Get output (no module-level references allowed)
                if use_input:
                    output = module_hook.input
                else:
                    output = module_hook.output

                # INLINE: Extract tensor from tuple if needed
                # For attn.output, it's already a tensor (not tuple)
                # For block output like h[0].output, it's a tuple
                saved = output[0].save() if isinstance(output, tuple) else output.save()
                model_out = model.lm_head.output.save()

            collected_activations[spec['key']] = saved

        return {
            'output': model_out,
            'activations': collected_activations,
        }

    # For vanilla and trainable interventions with sources
    source_activations = {}
    collected_activations = {}

    # Combine vanilla and trainable specs for processing
    all_intervention_specs = vanilla_specs + trainable_specs

    # Pre-compute which specs to apply by group (avoid iteration inside session)
    specs_by_group = {}
    for spec in all_intervention_specs:
        gid = spec['group_id']
        if gid not in specs_by_group:
            specs_by_group[gid] = []
        specs_by_group[gid].append(spec)

    with model.session(remote=True):
        # Collect source activations for all interventions (within session for value propagation)
        for group_id, specs_in_group in specs_by_group.items():
            if sources is None or group_id >= len(sources) or sources[group_id] is None:
                continue

            with model.trace(sources[group_id]):
                for spec in specs_in_group:
                    if spec.get('is_source_constant'):
                        continue  # Skip constant source interventions
                    module_hook = spec['module_hook']
                    use_input = (spec['hook_type'] == "input")
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    if isinstance(output, tuple):
                        source_activations[spec['key']] = output[0].save()
                    else:
                        source_activations[spec['key']] = output.save()

        # Apply interventions to base
        with model.trace(base, **kwargs):
            # Apply vanilla interventions (simple swap)
            for spec in vanilla_specs:
                if spec['key'] in source_activations:
                    module_hook = spec['module_hook']
                    use_input = (spec['hook_type'] == "input")
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    src = source_activations[spec['key']]
                    if isinstance(output, tuple):
                        output[0][:] = src
                    else:
                        output[:] = src

            # Apply trainable interventions (rotation-based)
            for spec in trainable_specs:
                if spec['key'] not in source_activations:
                    continue
                if not spec.get('intervention_weights'):
                    return NotImplementedError('Here have intervention weights decided')

                module_hook = spec['module_hook']
                use_input = (spec['hook_type'] == "input")
                if use_input:
                    base_output = module_hook.input
                else:
                    base_output = module_hook.output

                # Get base and source activations
                if isinstance(base_output, tuple):
                    base_act = base_output[0]
                else:
                    base_act = base_output
                source_act = source_activations[spec['key']]

                # Apply the trainable intervention using rotation logic
                weights = spec['intervention_weights']
                subspaces = spec.get('subspaces')
                intervention_type = weights.get('intervention_type', 'rotated_space')

                # Handle SigmoidMaskIntervention separately (no rotation)
                if intervention_type == 'sigmoid_mask':
                    # SigmoidMaskIntervention logic (no rotation)
                    mask = weights.get('mask').to(base_act.device)
                    temperature = weights.get('temperature')
                    mask_sigmoid = torch.sigmoid(mask / temperature)
                    intervened = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act

                else:
                    # All other trainable interventions use rotation
                    rotation_matrix = weights['rotate_layer_weight'].to(base_act.device)

                    # Project to rotated space
                    rotated_base = torch.matmul(base_act.to(rotation_matrix.dtype), rotation_matrix)
                    rotated_source = torch.matmul(source_act.to(rotation_matrix.dtype), rotation_matrix)
                    diff = rotated_source - rotated_base

                    if intervention_type == 'boundless_rotated_space':
                        # BoundlessRotatedSpaceIntervention logic
                        intervention_boundaries = weights.get('intervention_boundaries').to(base_act.device)
                        temperature = weights.get('temperature')
                        intervention_population = weights.get('intervention_population').to(base_act.device)
                        embed_dim = weights.get('embed_dim')
                        batch_size = base_act.shape[0]

                        intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
                        positions = intervention_population.repeat(batch_size, 1)
                        boundary_val = intervention_boundaries[0] * embed_dim
                        boundary_mask = torch.sigmoid(temperature * (boundary_val - positions))
                        boundary_mask = boundary_mask.to(rotated_base.dtype)

                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

                    elif intervention_type == 'sigmoid_mask_rotated_space':
                        # SigmoidMaskRotatedSpaceIntervention logic
                        masks = weights.get('masks').to(base_act.device)
                        temperature = weights.get('temperature')
                        batch_size = base_act.shape[0]

                        boundary_mask = torch.sigmoid(masks / temperature)
                        boundary_mask = torch.ones(batch_size, device=base_act.device).unsqueeze(dim=-1) * boundary_mask
                        boundary_mask = boundary_mask.to(rotated_base.dtype)

                        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
                        intervened = torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

                    elif subspaces is not None:
                        # Subspace handling for rotated_space and low_rank_rotated_space
                        subspace_partition = weights.get('subspace_partition')
                        use_fast = weights.get('use_fast', False)

                        can_use_fast = use_fast or (len(set(tuple(s) for s in subspaces)) == 1)
                        if can_use_fast:
                            if subspace_partition is None:
                                sel_subspace_indices = subspaces[0]
                            else:
                                sel_subspace_indices = []
                                for subspace in subspaces[0]:
                                    sel_subspace_indices.extend(subspace_partition[subspace])

                            batched_subspace = diff[..., sel_subspace_indices].unsqueeze(dim=1)
                            batched_weights = rotation_matrix[..., sel_subspace_indices].T
                            intervened = (base_act + torch.matmul(batched_subspace, batched_weights).squeeze(dim=1)).to(base_act.dtype)
                        else:
                            batched_subspace = []
                            batched_weights = []
                            for example_i in range(len(subspaces)):
                                sel_subspace_indices = []
                                for subspace in subspaces[example_i]:
                                    sel_subspace_indices.extend(subspace_partition[subspace])
                                LHS = diff[example_i, sel_subspace_indices].unsqueeze(dim=0)
                                RHS = rotation_matrix[..., sel_subspace_indices].T
                                batched_subspace.append(LHS)
                                batched_weights.append(RHS)
                            batched_subspace = torch.stack(batched_subspace, dim=0)
                            batched_weights = torch.stack(batched_weights, dim=0)
                            intervened = (base_act + torch.matmul(batched_subspace, batched_weights).squeeze(dim=1)).to(base_act.dtype)
                    else:
                        # Simple rotated_space or low_rank_rotated_space without subspaces
                        intervened = (base_act + torch.matmul(diff, rotation_matrix.T)).to(base_act.dtype)

                # Apply the intervened activation
                if isinstance(base_output, tuple):
                    base_output[0][:] = intervened
                else:
                    base_output[:] = intervened

            counterfactual_outputs = model.lm_head.output.save()

    return {
        'output': counterfactual_outputs,
        'activations': collected_activations,
    }


def execute_remote_generate(
    model,
    base,
    sources,
    intervention_specs,
    activations_sources=None,
    **kwargs
):
    """
    Execute generation with interventions on NDIF remote backend.

    Uses nnsight's model.generate() context for text generation.
    """
    # Identify which specs are collect vs vanilla
    collect_specs = [s for s in intervention_specs if s['is_collect']]
    vanilla_specs = [s for s in intervention_specs if s['is_vanilla']]

    # For collect interventions during generation
    if collect_specs and not vanilla_specs:
        collected_activations = {}

        # Generate with collection (one generate per spec for now)
        for spec in collect_specs:
            module_hook = spec['module_hook']
            hook_type = spec['hook_type']
            use_input = (hook_type == "input")

            with model.generate(base, remote=True, **kwargs):
                if use_input:
                    output = module_hook.input
                else:
                    output = module_hook.output

                saved = output[0].save() if isinstance(output, tuple) else output.save()
                gen_output = model.lm_head.output.save()

            collected_activations[spec['key']] = saved

        return {
            'output': gen_output,
            'activations': collected_activations,
        }

    # For vanilla interventions with sources during generation
    source_activations = {}
    collected_activations = {}

    # Collect source activations first
    if activations_sources is None and sources is not None:
        for spec in vanilla_specs:
            module_hook = spec['module_hook']
            hook_type = spec['hook_type']
            use_input = (hook_type == "input")
            source_input = sources[0] if len(sources) > 0 else None

            if source_input is not None:
                with model.trace(source_input, remote=True):
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    saved = output[0].save() if isinstance(output, tuple) else output.save()

                source_activations[spec['key']] = saved

    # Generate with interventions applied
    with model.session(remote=True):
        # Collect sources if needed within session
        if sources is not None and len(sources) > 0:
            with model.trace(sources[0]):
                for spec in vanilla_specs:
                    module_hook = spec['module_hook']
                    use_input = (spec['hook_type'] == "input")
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    if isinstance(output, tuple):
                        source_activations[spec['key']] = output[0].save()
                    else:
                        source_activations[spec['key']] = output.save()

        # Generate with interventions
        with model.generate(base, **kwargs):
            for spec in vanilla_specs:
                if spec['key'] in source_activations:
                    module_hook = spec['module_hook']
                    use_input = (spec['hook_type'] == "input")
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    src = source_activations[spec['key']]
                    if isinstance(output, tuple):
                        output[0][:] = src
                    else:
                        output[:] = src

            gen_output = model.lm_head.output.save()

    return {
        'output': gen_output,
        'activations': collected_activations,
    }
