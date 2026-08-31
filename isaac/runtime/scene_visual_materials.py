"""Shared matte visual materials for the Isaac UAV scenes.

FLOOR_COLOR：地板 RGB，範圍 0.0～1.0。數值越大越亮。
OBSTACLE_COLOR：障礙物 RGB。建議與地板保持明顯亮度或色相差異。
START_MARKER_COLOR：起點顏色。
GOAL_MARKER_COLOR：終點顏色。
MATERIAL_ROUGHNESS：粗糙度。1.0 最霧面，降低會增加高光。
MATERIAL_METALLIC：金屬度。維持 0.0 可避免金屬反射。
MATERIAL_SPECULAR_COLOR：鏡面反射顏色／強度。維持 (0, 0, 0) 最不反光。
DOME_LIGHT_INTENSITY：整體照明亮度。太暗可由 500 增至 600～800；過曝則降至 300～400。
DOME_LIGHT_COLOR：環境光顏色。維持 (1, 1, 1) 為中性白光。
DISABLE_ENVIRONMENT_LIGHTS：True 停用 Pegasus 原始 SphereLight；改成 False 會恢復原環境光。
RTX_SHADOWS_ENABLED：False 關閉陰影，True 恢復。
RTX_AMBIENT_OCCLUSION_ENABLED：False 關閉接觸暗影／AO，True 恢復。
"""

from __future__ import annotations


# RGB values are linear USD colors.  The light gray floor and navy obstacles
# provide both luminance and hue contrast in TOP and FPV images.
OBSTACLE_COLOR = (0.03, 0.08, 0.18)
FLOOR_COLOR = (0.60, 0.60, 0.60)
START_MARKER_COLOR = (0.0, 0.3, 1.0)
GOAL_MARKER_COLOR = (1.0, 0.0, 0.0)

MATERIAL_ROUGHNESS = 1.0
MATERIAL_METALLIC = 0.0
MATERIAL_SPECULAR_COLOR = (0.0, 0.0, 0.0)
MATERIAL_OPACITY = 1.0
MATERIAL_EMISSIVE_COLOR = (0.0, 0.0, 0.0)

# The Pegasus default environment contains a local 100000-intensity
# SphereLight.  Disable environment lights and replace them with one neutral,
# direction-independent DomeLight for the formal ML scene.
DISABLE_ENVIRONMENT_LIGHTS = True
DOME_LIGHT_INTENSITY = 800.0
DOME_LIGHT_COLOR = (1.0, 1.0, 1.0)
RTX_SHADOWS_ENABLED = False
RTX_AMBIENT_OCCLUSION_ENABLED = False

OBSTACLE_MATERIAL_NAME = "ObstacleMatte"
FLOOR_MATERIAL_NAME = "FloorMatte"
START_MARKER_MATERIAL_NAME = "StartMarkerMatte"
GOAL_MARKER_MATERIAL_NAME = "GoalMarkerMatte"


def create_matte_material(stage, path: str, color):
    """Create one texture-free UsdPreviewSurface matte material."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_ROUGHNESS))
    shader.CreateInput(
        "metallic", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_METALLIC))
    shader.CreateInput(
        "useSpecularWorkflow", Sdf.ValueTypeNames.Int
    ).Set(1)
    shader.CreateInput(
        "specularColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, MATERIAL_SPECULAR_COLOR)))
    shader.CreateInput(
        "opacity", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_OPACITY))
    shader.CreateInput(
        "emissiveColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, MATERIAL_EMISSIVE_COLOR)))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def create_scene_materials(stage, scene_root: str):
    """Create the shared scene materials once below ``scene_root``."""
    materials_root = f"{scene_root}/Materials"
    return {
        "obstacle": create_matte_material(
            stage,
            f"{materials_root}/{OBSTACLE_MATERIAL_NAME}",
            OBSTACLE_COLOR,
        ),
        "floor": create_matte_material(
            stage,
            f"{materials_root}/{FLOOR_MATERIAL_NAME}",
            FLOOR_COLOR,
        ),
        "start_marker": create_matte_material(
            stage,
            f"{materials_root}/{START_MARKER_MATERIAL_NAME}",
            START_MARKER_COLOR,
        ),
        "goal_marker": create_matte_material(
            stage,
            f"{materials_root}/{GOAL_MARKER_MATERIAL_NAME}",
            GOAL_MARKER_COLOR,
        ),
    }


def bind_material(prim, material) -> None:
    """Bind a shared USD material without changing geometry or physics APIs."""
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
