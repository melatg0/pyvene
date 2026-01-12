"""
Pure nnsight helper for remote execution.
This module intentionally does NOT import pyvene to avoid whitelist issues.
"""

CONST_INPUT_HOOK = "input"
CONST_OUTPUT_HOOK = "output"


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
    """
    # Identify which specs are collect vs vanilla
    collect_specs = [s for s in intervention_specs if s['is_collect']]
    vanilla_specs = [s for s in intervention_specs if s['is_vanilla']]


    # For collect interventions, collect each one separately
    if collect_specs and not vanilla_specs:
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
                model_out = model.output.save()

            collected_activations[spec['key']] = saved

        return {
            'output': model_out,
            'activations': collected_activations,
        }

    # For vanilla interventions with sources
    source_activations = {}
    collected_activations = {}

    # Collect sources first (one trace per source)
    if activations_sources is None and sources is not None:
        for group_id, keys in intervention_group.items():
            if sources[group_id] is None:
                continue

            for spec in vanilla_specs:
                if spec['group_id'] != group_id:
                    continue

                module_hook = spec['module_hook']
                hook_type = spec['hook_type']

                use_input = (hook_type == "input")

                with model.trace(sources[group_id], remote=True):
                    if use_input:
                        output = module_hook.input
                    else:
                        output = module_hook.output
                    saved = output[0].save() if isinstance(output, tuple) else output.save()

                source_activations[spec['key']] = saved

    # Apply vanilla interventions using session
    # Pre-compute which specs to apply (avoid iteration inside session)
    vanilla_by_group = {}
    for spec in vanilla_specs:
        gid = spec['group_id']
        if gid not in vanilla_by_group:
            vanilla_by_group[gid] = []
        vanilla_by_group[gid].append(spec)

    with model.session(remote=True):
        # Collect sources if needed (within session for value propagation)
        for group_id, specs_in_group in vanilla_by_group.items():
            if sources is None or sources[group_id] is None:
                continue

            with model.trace(sources[group_id]):
                for spec in specs_in_group:
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

        # Apply to base
        with model.trace(base, **kwargs):
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

            counterfactual_outputs = model.output.save()

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
                gen_output = model.output.save()

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

            gen_output = model.output.save()

    return {
        'output': gen_output,
        'activations': collected_activations,
    }
