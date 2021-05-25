from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2
from . import tfutil
from .tfutil import TfExpression
from .tfutil import TfExpressionEx

_dtype = tf.float64
_vars = OrderedDict() 
_immediate = OrderedDict() 
_finalized = False
_merge_op = None

def _create_var(name: str, value_expr: TfExpression) -> TfExpression:
    assert not _finalized
    name_id = name.replace("/", "_")
    v = tf.cast(value_expr, _dtype)

    if v.shape.is_fully_defined():
        size = np.prod(tfutil.shape_to_list(v.shape))
        size_expr = tf.constant(size, dtype=_dtype)
    else:
        size = None
        size_expr = tf.reduce_prod(tf.cast(tf.shape(v), _dtype))

    if size == 1:
        if v.shape.ndims != 0:
            v = tf.reshape(v, [])
        v = [size_expr, v, tf.square(v)]
    else:
        v = [size_expr, tf.reduce_sum(v), tf.reduce_sum(tf.square(v))]
    v = tf.cond(tf.is_finite(v[1]), lambda: tf.stack(v), lambda: tf.zeros(3, dtype=_dtype))

    with tfutil.absolute_name_scope("Autosummary/" + name_id), tf.control_dependencies(None):
        var = tf.Variable(tf.zeros(3, dtype=_dtype), trainable=False)  # [sum(1), sum(x), sum(x**2)]
    update_op = tf.cond(tf.is_variable_initialized(var), lambda: tf.assign_add(var, v), lambda: tf.assign(var, v))

    if name in _vars:
        _vars[name].append(var)
    else:
        _vars[name] = [var]
    return update_op


def autosummary(name: str, value: TfExpressionEx, passthru: TfExpressionEx = None) -> TfExpressionEx:
    tfutil.assert_tf_initialized()
    name_id = name.replace("/", "_")

    if tfutil.is_tf_expression(value):
        with tf.name_scope("summary_" + name_id), tf.device(value.device):
            update_op = _create_var(name, value)
            with tf.control_dependencies([update_op]):
                return tf.identity(value if passthru is None else passthru)

    else:  # python scalar or numpy array
        if name not in _immediate:
            with tfutil.absolute_name_scope("Autosummary/" + name_id), tf.device(None), tf.control_dependencies(None):
                update_value = tf.placeholder(_dtype)
                update_op = _create_var(name, update_value)
                _immediate[name] = update_op, update_value

        update_op, update_value = _immediate[name]
        tfutil.run(update_op, {update_value: value})
        return value if passthru is None else passthru


def finalize_autosummaries() -> None:
    global _finalized
    tfutil.assert_tf_initialized()

    if _finalized:
        return None

    _finalized = True
    tfutil.init_uninitialized_vars([var for vars_list in _vars.values() for var in vars_list])

    # Create summary ops.
    with tf.device(None), tf.control_dependencies(None):
        for name, vars_list in _vars.items():
            name_id = name.replace("/", "_")
            with tfutil.absolute_name_scope("Autosummary/" + name_id):
                moments = tf.add_n(vars_list)
                moments /= moments[0]
                with tf.control_dependencies([moments]):  # read before resetting
                    reset_ops = [tf.assign(var, tf.zeros(3, dtype=_dtype)) for var in vars_list]
                    with tf.name_scope(None), tf.control_dependencies(reset_ops):  # reset before reporting
                        mean = moments[1]
                        std = tf.sqrt(moments[2] - tf.square(moments[1]))
                        tf.summary.scalar(name, mean)
                        tf.summary.scalar("xCustomScalars/" + name + "/margin_lo", mean - std)
                        tf.summary.scalar("xCustomScalars/" + name + "/margin_hi", mean + std)

    # Group by category and chart name.
    cat_dict = OrderedDict()
    for series_name in sorted(_vars.keys()):
        p = series_name.split("/")
        cat = p[0] if len(p) >= 2 else ""
        chart = "/".join(p[1:-1]) if len(p) >= 3 else p[-1]
        if cat not in cat_dict:
            cat_dict[cat] = OrderedDict()
        if chart not in cat_dict[cat]:
            cat_dict[cat][chart] = []
        cat_dict[cat][chart].append(series_name)

    # Setup custom_scalar layout.
    categories = []
    for cat_name, chart_dict in cat_dict.items():
        charts = []
        for chart_name, series_names in chart_dict.items():
            series = []
            for series_name in series_names:
                series.append(layout_pb2.MarginChartContent.Series(
                    value=series_name,
                    lower="xCustomScalars/" + series_name + "/margin_lo",
                    upper="xCustomScalars/" + series_name + "/margin_hi"))
            margin = layout_pb2.MarginChartContent(series=series)
            charts.append(layout_pb2.Chart(title=chart_name, margin=margin))
        categories.append(layout_pb2.Category(title=cat_name, chart=charts))
    layout = summary_lib.custom_scalar_pb(layout_pb2.Layout(category=categories))
    return layout

def save_summaries(file_writer, global_step=None):
    global _merge_op
    tfutil.assert_tf_initialized()

    if _merge_op is None:
        layout = finalize_autosummaries()
        if layout is not None:
            file_writer.add_summary(layout)
        with tf.device(None), tf.control_dependencies(None):
            _merge_op = tf.summary.merge_all()

    file_writer.add_summary(_merge_op.eval(), global_step)
