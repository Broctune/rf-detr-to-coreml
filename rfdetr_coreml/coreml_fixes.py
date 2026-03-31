"""
Monkey-patches for coremltools bugs that block RF-DETR conversion.

Bug 1: _cast() does `dtype(x.val)` where x.val is a shape-(1,) numpy array.
Bug 2: view() can't handle shape list with non-scalar Var elements.
Bug 3: meshgrid() rejects rank-2 constant inputs (e.g. shape (N,1)) that
       are semantically 1D. coremltools 9.0 tightened the check to `rank > 1`
       but its own constant folding can reshape 1D inputs to (N,1).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_applied = False


def apply_coremltools_patches() -> None:
    """Apply monkey-patches for coremltools bugs that block RF-DETR conversion."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        import coremltools.converters.mil.frontend.torch.ops as ct_ops
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs, Var
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError:
        logger.warning("coremltools not installed, skipping patches")
        return

    # --- Patch 1: _cast (numpy scalar bug) ---
    def patched_cast(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")

        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            if not isinstance(val, dtype):
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    ct_ops._cast = patched_cast
    logger.info("Patched coremltools _cast")

    # --- Patch 2: view (shape list with non-scalar Vars) ---
    from coremltools.converters.mil.frontend.torch.ops import ListVar

    original_view = ct_ops.view

    def patched_view(context, node):
        inputs = _get_inputs(context, node, expected=2)
        x = inputs[0]
        shape = inputs[1]

        if isinstance(shape, Var) and np.prod(shape.shape) == 0:
            assert np.prod(x.shape) <= 1, (
                "Reshape to empty shape works only for scalar and single-element tensor"
            )
            context.add(mb.identity(x=x, name=node.name))
            return

        if isinstance(shape, ListVar):
            length = mb.list_length(ls=shape)
            indices = mb.range_1d(start=0, end=length, step=1)
            shape = mb.list_gather(ls=shape, indices=indices)

        # Handle list of Vars — squeeze any 1D single-element Vars to scalars
        if isinstance(shape, list) and all(isinstance(dim, Var) for dim in shape):
            int_shape = []
            for i, size in enumerate(shape):
                s = size
                # Squeeze 1D shape-(1,) Vars to scalar
                if len(s.shape) > 0:
                    s = mb.squeeze(x=s, name=node.name + f"_dim{i}_squeeze")
                if s.dtype != types.int32:
                    s = mb.cast(x=s, dtype="int32", name=node.name + f"_dim{i}_cast")
                int_shape.append(s)
            shape = mb.concat(values=int_shape, axis=0, name=node.name + "_shape")

        if isinstance(shape, Var):
            if shape.dtype != types.int32:
                shape = mb.cast(x=shape, dtype="int32", name=node.name + "_shape_cast")

        view = mb.reshape(x=x, shape=shape, name=node.name)
        context.add(view)

    # Replace the registered op handler
    ct_ops.view = patched_view
    # Also need to update the op registry
    try:
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY
        for alias in ["view", "view_copy", "_unsafe_view", "reshape"]:
            if hasattr(_TORCH_OPS_REGISTRY, 'set_func_by_name'):
                _TORCH_OPS_REGISTRY.set_func_by_name(patched_view, alias)
            elif alias in _TORCH_OPS_REGISTRY:
                _TORCH_OPS_REGISTRY[alias] = patched_view
    except ImportError:
        pass

    logger.info("Patched coremltools view op")

    # --- Patch 3: meshgrid (non-1D constant inputs in coremltools >=9.0) ---
    # coremltools 9.0 rejects meshgrid inputs with rank > 1, but its own
    # constant folding can turn shape-(N,) into shape-(N,1). We squeeze
    # the constant inputs back to 1D before they reach the check.
    try:
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

        original_meshgrid_ref = [None]

        def patched_meshgrid(context, node):
            inputs = _get_inputs(context, node, expected=[1, 2])
            tensor_inputs = inputs[0]
            if any(t.rank > 1 for t in tensor_inputs):
                for i, t in enumerate(tensor_inputs):
                    if t.rank > 1 and t.can_be_folded_to_const():
                        squeezed_val = t.val.flatten()
                        new_const = mb.const(val=squeezed_val, name=node.name + f"_sq{i}")
                        context.add(new_const, t.name)
                        tensor_inputs[i] = new_const
            original_meshgrid_ref[0](context, node)

        if hasattr(_TORCH_OPS_REGISTRY, 'get_func'):
            original_meshgrid_ref[0] = _TORCH_OPS_REGISTRY.get_func("meshgrid")
        elif hasattr(_TORCH_OPS_REGISTRY, '__getitem__'):
            original_meshgrid_ref[0] = _TORCH_OPS_REGISTRY["meshgrid"]

        if original_meshgrid_ref[0] is not None:
            if hasattr(_TORCH_OPS_REGISTRY, 'set_func_by_name'):
                _TORCH_OPS_REGISTRY.set_func_by_name(patched_meshgrid, 'meshgrid')
                _TORCH_OPS_REGISTRY.set_func_by_name(patched_meshgrid, 'meshgrid.indexing')
            else:
                _TORCH_OPS_REGISTRY["meshgrid"] = patched_meshgrid
                if "meshgrid.indexing" in _TORCH_OPS_REGISTRY:
                    _TORCH_OPS_REGISTRY["meshgrid.indexing"] = patched_meshgrid
            logger.info("Patched coremltools meshgrid op")
    except (ImportError, KeyError):
        pass
