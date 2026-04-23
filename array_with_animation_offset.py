bl_info = {
    "name": "Array with Animation Offset",
    "author": "Julbogje",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Array Anim Offset",
    "description": "Create an array of objects with offset animation — credits go to Claude for Coding and ARC9 for the base plugin",
    "category": "Object",
}

import bpy
import math
import mathutils
from bpy.props import (
    StringProperty, IntProperty, FloatVectorProperty,
    BoolProperty, FloatProperty, EnumProperty,
)


# ---------------------------------------------------------------------------
# FCurve compatibility (Blender < 4.4 vs layered Action API in 4.4+/5.0)
# ---------------------------------------------------------------------------

def get_fcurves(action):
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    fcurves = []
    if hasattr(action, 'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, 'channelbags'):
                    for cb in strip.channelbags:
                        fcurves.extend(cb.fcurves)
    return fcurves


# ---------------------------------------------------------------------------
# Deep-copy helper (armature + children, retarget modifiers)
# ---------------------------------------------------------------------------

def deep_copy_object(base_obj, context):
    new_obj = base_obj.copy()
    if base_obj.animation_data and base_obj.animation_data.action:
        new_obj.animation_data.action = base_obj.animation_data.action.copy()
    context.collection.objects.link(new_obj)

    new_children = []
    for child in base_obj.children:
        new_child = child.copy()
        has_shape_keys = (
            child.type == 'MESH'
            and child.data
            and child.data.shape_keys
            and child.data.shape_keys.animation_data
            and child.data.shape_keys.animation_data.action
        )
        if has_shape_keys:
            new_child.data = child.data.copy()
        if child.animation_data and child.animation_data.action:
            new_child.animation_data.action = child.animation_data.action.copy()
        context.collection.objects.link(new_child)

        new_child.parent = new_obj
        new_child.parent_type = child.parent_type
        if child.parent_type == 'BONE':
            new_child.parent_bone = child.parent_bone
        new_child.matrix_parent_inverse = child.matrix_parent_inverse.copy()

        for mod in new_child.modifiers:
            if mod.type == 'ARMATURE' and mod.object == base_obj:
                mod.object = new_obj

        new_children.append(new_child)

    return new_obj, new_children


# ---------------------------------------------------------------------------
# Easing / frame-offset calculation
# ---------------------------------------------------------------------------

def compute_frame_offset(instance_index, count, base_frame_offset, easing, multiplier):
    """
    Return the actual frame shift for copy at slot `instance_index`.

    easing values:
      LINEAR       – constant step: offset = index * base * multiplier
      EASE_IN      – slow start, fast end  (quadratic)
      EASE_OUT     – fast start, slow end  (quadratic)
      EASE_IN_OUT  – smooth S-curve        (smoothstep)

    For easing modes the *total* span of frame offsets stays the same as
    LINEAR (base * (count-1) * multiplier); individual copies are remapped
    along that span by the chosen curve.  This keeps the animation length
    predictable regardless of easing mode.
    """
    if count <= 1:
        return 0.0

    total_copies = count - 1          # number of copies (base is slot 0)
    total_span   = base_frame_offset * total_copies * multiplier
    t = instance_index / total_copies  # normalised position 0..1

    if easing == 'LINEAR':
        remapped = t
    elif easing == 'EASE_IN':
        remapped = t * t
    elif easing == 'EASE_OUT':
        remapped = t * (2.0 - t)
    else:  # EASE_IN_OUT  (smoothstep)
        remapped = t * t * (3.0 - 2.0 * t)

    return total_span * remapped


# ---------------------------------------------------------------------------
# Keyframe offsetting
# ---------------------------------------------------------------------------

def apply_offset_to_action(action, loc_offset, rot_offset_rad, scale_offset,
                            frame_shift, instance_index):
    """
    Shift all keyframes by `frame_shift` frames (already computed by
    compute_frame_offset) and bake loc/rot/scale offsets into keyframe values.

    scale_offset: per-step multiplicative factor tuple (x, y, z).
                  Keyframe values are multiplied by factor^instance_index.
    """
    LOC = {0: loc_offset[0], 1: loc_offset[1], 2: loc_offset[2]}
    ROT = {0: rot_offset_rad[0], 1: rot_offset_rad[1], 2: rot_offset_rad[2]}
    SC  = {0: scale_offset[0],   1: scale_offset[1],   2: scale_offset[2]}

    for fcurve in get_fcurves(action):
        dp = fcurve.data_path
        ai = fcurve.array_index

        val_offset = 0.0
        is_scale = False
        if dp == 'location':
            val_offset = LOC.get(ai, 0.0)
        elif dp == 'rotation_euler':
            val_offset = ROT.get(ai, 0.0)
        elif dp == 'scale':
            is_scale = True

        scale_factor = SC.get(ai, 1.0) ** instance_index if is_scale else 1.0

        for kp in fcurve.keyframe_points:
            kp.co.x           += frame_shift
            kp.handle_left.x  += frame_shift
            kp.handle_right.x += frame_shift

            if is_scale:
                kp.co.y           *= scale_factor
                kp.handle_left.y  *= scale_factor
                kp.handle_right.y *= scale_factor
            elif val_offset != 0.0:
                kp.co.y           += val_offset
                kp.handle_left.y  += val_offset
                kp.handle_right.y += val_offset


def offset_shape_keys(new_children, frame_shift, instance_index):
    for new_child in new_children:
        if (new_child.type == 'MESH'
                and new_child.data
                and new_child.data.shape_keys
                and new_child.data.shape_keys.animation_data
                and new_child.data.shape_keys.animation_data.action):
            sk_action = new_child.data.shape_keys.animation_data.action.copy()
            new_child.data.shape_keys.animation_data.action = sk_action
            apply_offset_to_action(
                sk_action,
                (0, 0, 0), (0, 0, 0), (0, 0, 0),
                frame_shift, instance_index,
            )


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------

def place_line(obj, base_obj, i, p):
    """
    Line shape. Works for both the base object (i=0) and copies (i>0).
    Returns (loc_offset, rot_offset_rad, scale_offset) for keyframe baking.
    """
    loc_offset = mathutils.Vector((0.0, 0.0, 0.0))
    if p.use_relative_offset:
        rel = mathutils.Vector(p.relative_offset)
        rel.x *= base_obj.dimensions.x
        rel.y *= base_obj.dimensions.y
        rel.z *= base_obj.dimensions.z
        loc_offset += rel
    if p.use_constant_offset:
        loc_offset += mathutils.Vector(p.constant_offset)

    obj.location = base_obj.location + loc_offset * i

    rot_per_step = mathutils.Vector(p.rotation_offset)
    obj.rotation_euler = mathutils.Euler((
        base_obj.rotation_euler.x + rot_per_step.x * i,
        base_obj.rotation_euler.y + rot_per_step.y * i,
        base_obj.rotation_euler.z + rot_per_step.z * i,
    ))

    sc = mathutils.Vector(p.scale_offset)
    obj.scale = mathutils.Vector((
        base_obj.scale.x * (sc.x ** i),
        base_obj.scale.y * (sc.y ** i),
        base_obj.scale.z * (sc.z ** i),
    ))

    loc_bake = loc_offset * i
    rot_bake = (rot_per_step.x * i, rot_per_step.y * i, rot_per_step.z * i)
    return loc_bake, rot_bake, tuple(p.scale_offset)


def place_circle(obj, base_obj, i, count, p):
    """
    Circle shape. i=0 → base object placed at its natural circle position.
    COUNT mode:    full 360° divided equally by `count` total slots.
    DISTANCE mode: fixed angle step per slot, independent of count.
    """
    axis   = p.circle_axis
    radius = p.circle_radius

    if p.circle_count_method == 'COUNT':
        angle_step = (2 * math.pi) / count
    else:
        angle_step = math.radians(p.circle_distance)

    angle = angle_step * i

    if axis == 'X':
        offset      = mathutils.Vector((0.0,
                                        radius * (math.cos(angle) - 1.0),
                                        radius * math.sin(angle)))
        tangent_dir = mathutils.Vector((0.0, -math.sin(angle), math.cos(angle)))
        normal_dir  = mathutils.Vector((1.0, 0.0, 0.0))
    elif axis == 'Y':
        offset      = mathutils.Vector((radius * (math.cos(angle) - 1.0),
                                        0.0,
                                        radius * math.sin(angle)))
        tangent_dir = mathutils.Vector((-math.sin(angle), 0.0, math.cos(angle)))
        normal_dir  = mathutils.Vector((0.0, 1.0, 0.0))
    else:  # Z
        offset      = mathutils.Vector((radius * (math.cos(angle) - 1.0),
                                        radius * math.sin(angle),
                                        0.0))
        tangent_dir = mathutils.Vector((-math.sin(angle), math.cos(angle), 0.0))
        normal_dir  = mathutils.Vector((0.0, 0.0, 1.0))

    obj.location = base_obj.location + offset

    if p.circle_align_rotation:
        radial_dir = offset.normalized() if offset.length > 1e-6 else tangent_dir

        axis_vecs = {
            'X':  mathutils.Vector((1, 0, 0)),  'Y':  mathutils.Vector((0, 1, 0)),
            'Z':  mathutils.Vector((0, 0, 1)),  '-X': mathutils.Vector((-1, 0, 0)),
            '-Y': mathutils.Vector((0, -1, 0)), '-Z': mathutils.Vector((0, 0, -1)),
        }

        def circle_semantic(name):
            if name == 'TANGENT': return tangent_dir
            if name == 'NORMAL':  return normal_dir
            if name == 'RADIAL':  return radial_dir
            return axis_vecs.get(name, mathutils.Vector((0, 1, 0)))

        fwd = circle_semantic(p.circle_forward_axis).normalized()
        up  = circle_semantic(p.circle_up_axis).normalized()

        right = fwd.cross(up)
        if right.length < 1e-6:
            up    = mathutils.Vector((0, 0, 1))
            if abs(fwd.dot(up)) > 0.99:
                up = mathutils.Vector((1, 0, 0))
            right = fwd.cross(up)
        right.normalize()
        up_corrected = right.cross(fwd).normalized()

        rot_mat = mathutils.Matrix((right, fwd, up_corrected)).transposed().to_4x4()
        obj.rotation_euler = (rot_mat @ base_obj.rotation_euler.to_matrix().to_4x4()).to_euler()
    else:
        obj.rotation_euler = base_obj.rotation_euler.copy()

    return offset, (0, 0, 0), (0, 0, 0)


def _sample_curve(curve_obj, SAMPLES=512):
    """Return (points, lengths, total_length) for the first spline of curve_obj."""
    spline = curve_obj.data.splines[0]
    points = []
    for s in range(SAMPLES + 1):
        t = s / SAMPLES
        if spline.type == 'BEZIER':
            bpts     = spline.bezier_points
            seg_count = len(bpts) - 1
            if seg_count <= 0:
                pt = curve_obj.matrix_world @ bpts[0].co
            else:
                seg_f = t * seg_count
                seg_i = min(int(seg_f), seg_count - 1)
                seg_t = seg_f - seg_i
                p0 = bpts[seg_i].co
                p3 = bpts[seg_i + 1].co
                p1 = bpts[seg_i].handle_right
                p2 = bpts[seg_i + 1].handle_left
                u  = 1 - seg_t
                co = (u**3)*p0 + 3*(u**2)*seg_t*p1 + 3*u*(seg_t**2)*p2 + (seg_t**3)*p3
                pt = curve_obj.matrix_world @ co
        else:
            spts = spline.points
            idx  = min(int(t * len(spts)), len(spts) - 1)
            pt   = curve_obj.matrix_world @ spts[idx].co.xyz
        points.append(mathutils.Vector(pt))

    lengths = [0.0]
    for k in range(1, len(points)):
        lengths.append(lengths[-1] + (points[k] - points[k - 1]).length)
    return points, lengths, lengths[-1]


def place_curve(obj, base_obj, i, count, p, context):
    """
    Curve shape. i=0 → base placed at the start of the curve.
    COUNT mode:    (count-1) intervals span the full curve length.
    DISTANCE mode: each slot is curve_distance metres apart, count caps total.
    """
    curve_obj = bpy.data.objects.get(p.curve_object_name)
    if not curve_obj or curve_obj.type != 'CURVE':
        obj.location = base_obj.location
        return mathutils.Vector((0, 0, 0)), (0, 0, 0), (0, 0, 0)

    depsgraph  = context.evaluated_depsgraph_get()
    curve_eval = curve_obj.evaluated_get(depsgraph)

    points, lengths, total_length = _sample_curve(curve_eval)

    if total_length < 1e-6:
        obj.location = base_obj.location
        return mathutils.Vector((0, 0, 0)), (0, 0, 0), (0, 0, 0)

    if p.curve_count_method == 'COUNT':
        intervals   = max(count - 1, 1)
        target_dist = (total_length / intervals) * i
    else:
        target_dist = p.curve_distance * i

    target_dist = min(target_dist, total_length)

    pos     = points[-1].copy()
    tangent = (points[-1] - points[-2]).normalized()
    for k in range(1, len(lengths)):
        if lengths[k] >= target_dist:
            seg_len = lengths[k] - lengths[k - 1]
            seg_t   = (target_dist - lengths[k - 1]) / max(seg_len, 1e-9)
            pos     = points[k - 1].lerp(points[k], seg_t)
            tangent = (points[k] - points[k - 1]).normalized()
            break

    obj.location = pos

    if p.curve_align_rotation:
        world_up = mathutils.Vector((0, 0, 1))
        if abs(tangent.dot(world_up)) > 0.99:
            world_up = mathutils.Vector((1, 0, 0))
        right_default = tangent.cross(world_up).normalized()
        up_default    = right_default.cross(tangent).normalized()

        axis_vecs = {
            'X':  mathutils.Vector((1, 0, 0)),  'Y':  mathutils.Vector((0, 1, 0)),
            'Z':  mathutils.Vector((0, 0, 1)),  '-X': mathutils.Vector((-1, 0, 0)),
            '-Y': mathutils.Vector((0, -1, 0)), '-Z': mathutils.Vector((0, 0, -1)),
        }

        def curve_semantic(name):
            if name == 'TANGENT': return tangent
            if name == 'UP':      return up_default
            return axis_vecs.get(name, tangent)

        fwd = curve_semantic(p.curve_forward_axis).normalized()
        up  = curve_semantic(p.curve_up_axis).normalized()

        right = fwd.cross(up)
        if right.length < 1e-6:
            up    = world_up if abs(fwd.dot(world_up)) < 0.99 else mathutils.Vector((1, 0, 0))
            right = fwd.cross(up)
        right.normalize()
        up_corrected = right.cross(fwd).normalized()

        rot_mat = mathutils.Matrix((right, fwd, up_corrected)).transposed().to_4x4()
        obj.rotation_euler = rot_mat.to_euler()
    else:
        obj.rotation_euler = base_obj.rotation_euler.copy()

    loc_offset = pos - base_obj.location
    return loc_offset, (0, 0, 0), (0, 0, 0)


def place_transform(obj, base_obj, i, p):
    """
    Transform shape. i=0 → base object placed at its accumulated transform slot 0
    (i.e. no movement: loc_step*0=0, scale^0=1).
    """
    if p.transform_reference == 'OBJECT':
        ref_obj = bpy.data.objects.get(p.transform_object_name)
        if ref_obj:
            loc_step = (ref_obj.location - base_obj.location
                        if p.transform_relative_space else ref_obj.location.copy())
            rot_step_rad = (ref_obj.rotation_euler.x,
                            ref_obj.rotation_euler.y,
                            ref_obj.rotation_euler.z)
            sc_step = mathutils.Vector(ref_obj.scale)
        else:
            loc_step     = mathutils.Vector((0, 0, 0))
            rot_step_rad = (0, 0, 0)
            sc_step      = mathutils.Vector((1, 1, 1))
    else:
        loc_step = mathutils.Vector(p.transform_location)
        rot_step_rad = (
            math.radians(p.transform_rotation[0]),
            math.radians(p.transform_rotation[1]),
            math.radians(p.transform_rotation[2]),
        )
        sc_step = mathutils.Vector(p.transform_scale)

    loc_off     = loc_step * i
    rot_off_rad = (rot_step_rad[0] * i, rot_step_rad[1] * i, rot_step_rad[2] * i)

    obj.location = base_obj.location + loc_off
    obj.rotation_euler = mathutils.Euler((
        base_obj.rotation_euler.x + rot_off_rad[0],
        base_obj.rotation_euler.y + rot_off_rad[1],
        base_obj.rotation_euler.z + rot_off_rad[2],
    ))
    obj.scale = mathutils.Vector((
        base_obj.scale.x * (sc_step.x ** i),
        base_obj.scale.y * (sc_step.y ** i),
        base_obj.scale.z * (sc_step.z ** i),
    ))

    return loc_off, rot_off_rad, tuple(sc_step)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class OBJECT_OT_animated_array(bpy.types.Operator):
    bl_idname = "object.animated_array"
    bl_label  = "Create Animated Array"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        p     = scene.animated_array_params

        base_object = bpy.data.objects.get(p.base_object_name)
        if not base_object:
            self.report({'ERROR'}, f"No object named '{p.base_object_name}' found")
            return {'CANCELLED'}
        if not base_object.animation_data or not base_object.animation_data.action:
            self.report({'ERROR'}, "Base object has no animation data or action")
            return {'CANCELLED'}

        shape = p.array_shape
        count = p.count   # TOTAL objects including base (slot 0)

        # ---- Apply transform to the base object itself (slot i=0) ----
        # For LINE:      i=0 → no movement (loc_offset*0=0, scale^0=1) — base stays put ✓
        # For CIRCLE:    i=0 → angle=0 → offset=(cos(0)-1, sin(0))=(0,0) — base stays put ✓
        # For CURVE:     i=0 → target_dist=0 → base moves to curve start
        # For TRANSFORM: i=0 → loc_step*0=0 — base stays put ✓
        #
        # The base object does NOT get frame-offset applied (its index = 0 → shift = 0).
        if shape == 'LINE':
            place_line(base_object, base_object, 0, p)
        elif shape == 'CIRCLE':
            place_circle(base_object, base_object, 0, count, p)
        elif shape == 'CURVE':
            place_curve(base_object, base_object, 0, count, p, context)
        elif shape == 'TRANSFORM':
            place_transform(base_object, base_object, 0, p)

        # ---- Create and place copies (slots i=1 .. count-1) ----
        for i in range(1, count):
            new_obj, new_children = deep_copy_object(base_object, context)

            if shape == 'LINE':
                loc_off, rot_off, sc_off = place_line(new_obj, base_object, i, p)
            elif shape == 'CIRCLE':
                loc_off, rot_off, sc_off = place_circle(new_obj, base_object, i, count, p)
            elif shape == 'CURVE':
                loc_off, rot_off, sc_off = place_curve(new_obj, base_object, i, count, p, context)
            elif shape == 'TRANSFORM':
                loc_off, rot_off, sc_off = place_transform(new_obj, base_object, i, p)
            else:
                loc_off, rot_off, sc_off = (0, 0, 0), (0, 0, 0), (0, 0, 0)

            frame_shift = compute_frame_offset(
                i, count, p.frame_offset, p.frame_easing, p.frame_multiplier
            )

            if new_obj.animation_data and new_obj.animation_data.action:
                apply_offset_to_action(
                    new_obj.animation_data.action,
                    loc_off, rot_off, sc_off,
                    frame_shift, i,
                )

            offset_shape_keys(new_children, frame_shift, i)

        self.report({'INFO'}, f"Created {count - 1} copies (shape: {shape})")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class AnimatedArrayProperties(bpy.types.PropertyGroup):

    base_object_name: StringProperty(name="Base Object", default="Cube")
    count: IntProperty(name="Count", default=5, min=2,
        description="Total number of objects including the original (base = slot 0)")

    # Frame offset
    frame_offset: FloatProperty(
        name="Frame Offset", default=1.0, min=0.0,
        description="Base frame offset between each copy's animation")
    frame_multiplier: FloatProperty(
        name="Multiplier", default=1.0, min=0.0,
        description="Multiplies the base frame offset (e.g. 2.0 = double spacing)")
    frame_easing: EnumProperty(
        name="Easing",
        items=[
            ('LINEAR',      "Linear",       "Constant frame offset between each copy"),
            ('EASE_IN',     "Ease In",      "Copies start close together and spread apart"),
            ('EASE_OUT',    "Ease Out",     "Copies start spread apart and come together"),
            ('EASE_IN_OUT', "Ease In/Out",  "S-curve: slow at both ends, fast in the middle"),
        ],
        default='LINEAR',
        description="How the frame offset is distributed across copies",
    )

    array_shape: EnumProperty(
        name="Shape",
        items=[
            ('LINE',      "Line",      "Distribute in a line with position/rotation/scale offsets"),
            ('CIRCLE',    "Circle",    "Distribute around a circle"),
            ('CURVE',     "Curve",     "Distribute along a curve object"),
            ('TRANSFORM', "Transform", "Accumulate a transform from an object or manual inputs"),
        ],
        default='LINE',
    )

    # Line
    use_relative_offset: BoolProperty(name="Use Relative Offset", default=True)
    relative_offset: FloatVectorProperty(name="Relative Offset", default=(1.0, 0.0, 0.0), size=3)
    use_constant_offset: BoolProperty(name="Use Constant Offset", default=False)
    constant_offset: FloatVectorProperty(name="Constant Offset", default=(0.0, 0.0, 0.0), size=3)
    rotation_offset: FloatVectorProperty(
        name="Rotation Offset", default=(0.0, 0.0, 0.0), size=3, subtype='EULER',
        description="Rotation added per copy (radians)")
    scale_offset: FloatVectorProperty(
        name="Scale Offset", default=(1.0, 1.0, 1.0), size=3,
        description="Scale multiplied per copy (1.0 = no change, 1.1 = 10% bigger each step)")

    # Circle
    circle_count_method: EnumProperty(
        name="Count Method",
        items=[('COUNT',    "Count",    "Fixed number of copies filling 360°"),
               ('DISTANCE', "Distance", "Fixed angle in degrees between copies")],
        default='COUNT',
    )
    circle_axis: EnumProperty(
        name="Central Axis",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='Z',
    )
    circle_radius: FloatProperty(name="Radius", default=2.0, min=0.0)
    circle_distance: FloatProperty(name="Angle (degrees)", default=30.0, min=0.1, max=360.0,
        description="Angle in degrees between each copy")
    circle_align_rotation: BoolProperty(name="Align Rotation", default=True)
    circle_forward_axis: EnumProperty(
        name="Forward Axis",
        items=[
            ('TANGENT', "Tangent", "Point along the circle tangent (travel direction)"),
            ('NORMAL',  "Normal",  "Point along the circle's central axis"),
            ('RADIAL',  "Radial",  "Point away from the circle centre"),
            ('X', "X", "World +X"), ('Y', "Y", "World +Y"), ('Z', "Z", "World +Z"),
            ('-X', "-X", "World -X"), ('-Y', "-Y", "World -Y"), ('-Z', "-Z", "World -Z"),
        ],
        default='TANGENT',
    )
    circle_up_axis: EnumProperty(
        name="Up Axis",
        items=[
            ('NORMAL',  "Normal",  "Up along the circle's central axis"),
            ('RADIAL',  "Radial",  "Up pointing away from the circle centre"),
            ('TANGENT', "Tangent", "Up along the tangent"),
            ('X', "X", "World +X"), ('Y', "Y", "World +Y"), ('Z', "Z", "World +Z"),
            ('-X', "-X", "World -X"), ('-Y', "-Y", "World -Y"), ('-Z', "-Z", "World -Z"),
        ],
        default='NORMAL',
    )

    # Curve
    curve_count_method: EnumProperty(
        name="Count Method",
        items=[('COUNT',    "Count",    "Divide curve length evenly across all objects"),
               ('DISTANCE', "Distance", "Place every N metres along the curve")],
        default='COUNT',
    )
    curve_object_name: StringProperty(name="Curve Object", default="")
    curve_distance: FloatProperty(name="Distance (m)", default=1.0, min=0.001)
    curve_relative_space: BoolProperty(name="Relative Space", default=True)
    curve_align_rotation: BoolProperty(name="Align Rotation", default=True)
    curve_forward_axis: EnumProperty(
        name="Forward Axis",
        items=[
            ('TANGENT', "Tangent", "Point along the curve tangent"),
            ('X', "X", "World +X"), ('Y', "Y", "World +Y"), ('Z', "Z", "World +Z"),
            ('-X', "-X", "World -X"), ('-Y', "-Y", "World -Y"), ('-Z', "-Z", "World -Z"),
        ],
        default='TANGENT',
    )
    curve_up_axis: EnumProperty(
        name="Up Axis",
        items=[
            ('UP', "Curve Up", "Use the curve's computed up direction"),
            ('X', "X", "World +X"), ('Y', "Y", "World +Y"), ('Z', "Z", "World +Z"),
            ('-X', "-X", "World -X"), ('-Y', "-Y", "World -Y"), ('-Z', "-Z", "World -Z"),
        ],
        default='Z',
    )

    # Transform
    transform_reference: EnumProperty(
        name="Transform Reference",
        items=[('INPUTS', "Inputs", "Use manual values below"),
               ('OBJECT', "Object", "Read transform from a reference object")],
        default='INPUTS',
    )
    transform_object_name: StringProperty(name="Transform Object", default="")
    transform_relative_space: BoolProperty(name="Relative Space", default=True)
    transform_location: FloatVectorProperty(name="Location", default=(0.0, 0.0, 0.0), size=3)
    transform_rotation: FloatVectorProperty(name="Rotation (degrees)", default=(0.0, 0.0, 0.0), size=3)
    transform_scale: FloatVectorProperty(name="Scale", default=(1.0, 1.0, 1.0), size=3,
        description="Scale multiplied per copy (1.0 = no change)")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class OBJECT_PT_animated_array(bpy.types.Panel):
    bl_label      = "Animated Array"
    bl_idname     = "OBJECT_PT_animated_array"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category   = 'Array Anim Offset'

    def draw(self, context):
        layout = self.layout
        p      = context.scene.animated_array_params
        shape  = p.array_shape

        layout.prop(p, "base_object_name")

        # ---- Frame offset section ----
        box = layout.box()
        box.label(text="Frame Offset")
        row = box.row(align=True)
        row.prop(p, "frame_offset")
        row.prop(p, "frame_multiplier", text="×")
        box.prop(p, "frame_easing", text="Easing")

        layout.prop(p, "array_shape")
        layout.separator()

        # ---- Shape-specific settings ----
        if shape == 'LINE':
            layout.prop(p, "count")

            box = layout.box()
            box.prop(p, "use_relative_offset")
            if p.use_relative_offset:
                box.prop(p, "relative_offset")

            box = layout.box()
            box.prop(p, "use_constant_offset")
            if p.use_constant_offset:
                box.prop(p, "constant_offset")

            box = layout.box()
            box.label(text="Rotation Offset (per copy, radians)")
            box.prop(p, "rotation_offset", text="")

            box = layout.box()
            box.label(text="Scale Offset (per copy, multiplicative)")
            box.prop(p, "scale_offset", text="")

        elif shape == 'CIRCLE':
            row = layout.row()
            row.prop(p, "circle_count_method", expand=True)
            if p.circle_count_method == 'COUNT':
                layout.prop(p, "count")
            else:
                layout.prop(p, "circle_distance")
            row = layout.row()
            row.label(text="Central Axis")
            row.prop(p, "circle_axis", expand=True)
            layout.prop(p, "circle_radius")
            col = layout.column()
            col.prop(p, "circle_align_rotation")
            if p.circle_align_rotation:
                box = col.box()
                row = box.row()
                row.label(text="Forward Axis")
                row.prop(p, "circle_forward_axis", expand=True)
                row = box.row()
                row.label(text="Up Axis")
                row.prop(p, "circle_up_axis", expand=True)

        elif shape == 'CURVE':
            row = layout.row()
            row.prop(p, "curve_count_method", expand=True)
            if p.curve_count_method == 'COUNT':
                layout.prop(p, "count")
            else:
                layout.prop(p, "curve_distance")
                layout.prop(p, "count", text="Max Copies")
            layout.prop(p, "curve_object_name")
            layout.prop(p, "curve_relative_space")
            col = layout.column()
            col.prop(p, "curve_align_rotation")
            if p.curve_align_rotation:
                box = col.box()
                row = box.row()
                row.label(text="Forward Axis")
                row.prop(p, "curve_forward_axis", expand=True)
                row = box.row()
                row.label(text="Up Axis")
                row.prop(p, "curve_up_axis", expand=True)

        elif shape == 'TRANSFORM':
            layout.prop(p, "count")
            row = layout.row()
            row.label(text="Transform Reference")
            row.prop(p, "transform_reference", expand=True)
            if p.transform_reference == 'OBJECT':
                layout.prop(p, "transform_object_name")
                layout.prop(p, "transform_relative_space")
            else:
                box = layout.box()
                box.prop(p, "transform_location")
                box.prop(p, "transform_rotation")
                box.prop(p, "transform_scale")

        layout.separator()
        layout.operator("object.animated_array")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    OBJECT_OT_animated_array,
    AnimatedArrayProperties,
    OBJECT_PT_animated_array,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.animated_array_params = bpy.props.PointerProperty(
        type=AnimatedArrayProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.animated_array_params


if __name__ == "__main__":
    register()
