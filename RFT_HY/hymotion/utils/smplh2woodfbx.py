import glob
import os
import shutil
import tempfile
from typing import Dict, Optional

import fbx
import numpy as np
import torch
from transforms3d.euler import mat2euler

from .geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_angle_axis

# yapf: disable
SMPLH_JOINT2NUM = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
    "L_Ankle": 7, "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11,
    "Neck": 12, "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17,
    "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "L_Index1": 22, "L_Index2": 23, "L_Index3": 24,
    "L_Middle1": 25, "L_Middle2": 26, "L_Middle3": 27,
    "L_Pinky1": 28, "L_Pinky2": 29, "L_Pinky3": 30,
    "L_Ring1": 31, "L_Ring2": 32, "L_Ring3": 33,
    "L_Thumb1": 34, "L_Thumb2": 35, "L_Thumb3": 36,
    "R_Index1": 37, "R_Index2": 38, "R_Index3": 39,
    "R_Middle1": 40, "R_Middle2": 41, "R_Middle3": 42,
    "R_Pinky1": 43, "R_Pinky2": 44, "R_Pinky3": 45,
    "R_Ring1": 46, "R_Ring2": 47, "R_Ring3": 48,
    "R_Thumb1": 49, "R_Thumb2": 50, "R_Thumb3": 51,
}

# Mapping from SMPL-H joint names to lowercase names used in some FBX templates
SMPLH_TO_LOWERCASE_MAPPING = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "Spine1": "spine1",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "Spine2": "spine2",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "Spine3": "spine3",
    "L_Foot": "left_foot",
    "R_Foot": "right_foot",
    "Neck": "neck",
    "L_Collar": "left_collar",
    "R_Collar": "right_collar",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
    "L_Index1": "left_index1",
    "L_Index2": "left_index2",
    "L_Index3": "left_index3",
    "L_Middle1": "left_middle1",
    "L_Middle2": "left_middle2",
    "L_Middle3": "left_middle3",
    "L_Pinky1": "left_pinky1",
    "L_Pinky2": "left_pinky2",
    "L_Pinky3": "left_pinky3",
    "L_Ring1": "left_ring1",
    "L_Ring2": "left_ring2",
    "L_Ring3": "left_ring3",
    "L_Thumb1": "left_thumb1",
    "L_Thumb2": "left_thumb2",
    "L_Thumb3": "left_thumb3",
    "R_Index1": "right_index1",
    "R_Index2": "right_index2",
    "R_Index3": "right_index3",
    "R_Middle1": "right_middle1",
    "R_Middle2": "right_middle2",
    "R_Middle3": "right_middle3",
    "R_Pinky1": "right_pinky1",
    "R_Pinky2": "right_pinky2",
    "R_Pinky3": "right_pinky3",
    "R_Ring1": "right_ring1",
    "R_Ring2": "right_ring2",
    "R_Ring3": "right_ring3",
    "R_Thumb1": "right_thumb1",
    "R_Thumb2": "right_thumb2",
    "R_Thumb3": "right_thumb3",
}
# yapf: enable


def _loadFbxScene(fbxManager, filepath):
    """Load an FBX file into a scene"""
    importer = fbx.FbxImporter.Create(fbxManager, "")

    if not importer.Initialize(filepath, -1, fbxManager.GetIOSettings()):
        raise Exception(
            f"Failed to initialize FBX importer for: {filepath}\nError: {importer.GetStatus().GetErrorString()}"
        )

    fbxScene = fbx.FbxScene.Create(fbxManager, "")
    importer.Import(fbxScene)
    importer.Destroy()

    return fbxScene


def _collectAllNodes(node, nodes_dict=None):
    """Recursively collect all nodes in the scene hierarchy"""
    if nodes_dict is None:
        nodes_dict = {}

    nodes_dict[node.GetName()] = node

    for i in range(node.GetChildCount()):
        _collectAllNodes(node.GetChild(i), nodes_dict)

    return nodes_dict


def _collectSkeletonNodes(node, skeleton_nodes=None):
    """Recursively collect skeleton/bone nodes"""
    if skeleton_nodes is None:
        skeleton_nodes = {}

    # Check if this node has a skeleton attribute
    attr = node.GetNodeAttribute()
    if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
        skeleton_nodes[node.GetName()] = node

    for i in range(node.GetChildCount()):
        _collectSkeletonNodes(node.GetChild(i), skeleton_nodes)

    return skeleton_nodes


def _animateSingleChannel(animLayer, component, name, values, frameDuration):
    """Animate a single channel (X, Y, or Z) with keyframes"""
    ncomp = {"X": 0, "Y": 1, "Z": 2}.get(name, 0)

    time = fbx.FbxTime()
    curve = component.GetCurve(animLayer, name, True)
    curve.KeyModifyBegin()
    for nth in range(len(values)):
        time.SetSecondDouble(nth * frameDuration)
        keyIndex = curve.KeyAdd(time)[0]
        curve.KeySetValue(keyIndex, values[nth][ncomp])
        curve.KeySetInterpolation(keyIndex, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
    curve.KeyModifyEnd()


def _animateRotationKeyFrames(animLayer, node, rot_matrices, frameDuration):
    """Animate rotation keyframes for a node using rotation matrices"""
    rotations = []
    for nth in range(len(rot_matrices)):
        # Convert rotation matrix to Euler angles (XYZ order)
        euler = np.rad2deg(mat2euler(rot_matrices[nth], axes="sxyz"))
        rotations.append(euler)

    _animateSingleChannel(animLayer, node.LclRotation, "X", rotations, frameDuration)
    _animateSingleChannel(animLayer, node.LclRotation, "Y", rotations, frameDuration)
    _animateSingleChannel(animLayer, node.LclRotation, "Z", rotations, frameDuration)


def _animateTranslationKeyFrames(animLayer, node, translations, frameDuration):
    """Animate translation keyframes for a node"""
    # Ensure translations is a numpy array with shape (num_frames, 3)
    if isinstance(translations, torch.Tensor):
        translations = translations.numpy()
    translations = np.asarray(translations, dtype=np.float64)

    if len(translations.shape) == 1:
        # Single frame, reshape to (1, 3)
        translations = translations.reshape(1, -1)

    _animateSingleChannel(animLayer, node.LclTranslation, "X", translations, frameDuration)
    _animateSingleChannel(animLayer, node.LclTranslation, "Y", translations, frameDuration)
    _animateSingleChannel(animLayer, node.LclTranslation, "Z", translations, frameDuration)


def _clearExistingAnimations(fbxScene):
    """Remove all existing animation stacks from the scene"""
    anim_stack_count = fbxScene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    for i in range(anim_stack_count - 1, -1, -1):
        anim_stack = fbxScene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
        if anim_stack:
            anim_stack.Destroy()


def _applyAnimationToSkeleton(fbxScene, nodes_map, rot_matrices, translations, fps, smplh_to_fbx_mapping, name="Take1"):
    """
    Apply SMPL-H animation data to skeleton nodes in the FBX scene.

    Args:
        fbxScene: FBX scene object
        nodes_map: Dictionary of node_name -> FbxNode
        rot_matrices: (num_frames, num_joints, 3, 3) rotation matrices
        translations: (num_frames, 3) root translations (relative displacement, not absolute position)
        fps: Frame rate
        smplh_to_fbx_mapping: Mapping from SMPL-H joint names to FBX node names
        name: Animation take name
    """
    frameDuration = 1.0 / fps
    num_frames = rot_matrices.shape[0]
    num_joints = rot_matrices.shape[1]

    # Create animation stack and layer
    animStack = fbx.FbxAnimStack.Create(fbxScene, name)
    animLayer = fbx.FbxAnimLayer.Create(fbxScene, "Base Layer")
    animStack.AddMember(animLayer)

    # Track if root translation was applied
    root_translation_applied = False
    root_node = None

    # Get root node's initial LclTranslation from template (this is like Translates[0] in smplh2woodfbx.py)
    root_initial_translation = None
    root_fbx_name = smplh_to_fbx_mapping.get("Pelvis")
    if root_fbx_name and root_fbx_name in nodes_map:
        root_node_temp = nodes_map[root_fbx_name]
        initial_trans = root_node_temp.LclTranslation.Get()
        root_initial_translation = np.array([initial_trans[0], initial_trans[1], initial_trans[2]])
        print(f"Root initial LclTranslation from template: {root_initial_translation}")

    # Animate each joint
    for smplh_joint_name, smplh_joint_idx in SMPLH_JOINT2NUM.items():
        if smplh_joint_idx >= num_joints:
            continue

        # Get the FBX node name from mapping
        fbx_node_name = smplh_to_fbx_mapping.get(smplh_joint_name)
        if not fbx_node_name:
            if smplh_joint_idx == 0:
                print(f"Warning: Root joint 'Pelvis' not found in mapping!")
            continue

        # Find the node
        node = nodes_map.get(fbx_node_name)
        if not node:
            print(f"Warning: Joint '{smplh_joint_name}' (FBX: '{fbx_node_name}') not found in scene")
            continue

        # Animate rotation
        _animateRotationKeyFrames(
            animLayer=animLayer,
            node=node,
            rot_matrices=rot_matrices[:, smplh_joint_idx],
            frameDuration=frameDuration,
        )

        # Animate translation for root joint (Pelvis)
        if smplh_joint_idx == 0:
            root_node = node
            # Add initial offset to translations (like smplh2woodfbx.py does: Translates[0] + trans)
            # The translations input is relative displacement, we need to add the template's initial position
            if root_initial_translation is not None:
                final_translations = translations + root_initial_translation
                print(
                    f"Applying root translation to '{fbx_node_name}', frames={num_frames}, "
                    f"initial_offset={root_initial_translation}, "
                    f"final translation range: {final_translations.min(axis=0)} to {final_translations.max(axis=0)}"
                )
            else:
                final_translations = translations
                print(
                    f"Applying root translation to '{fbx_node_name}', frames={num_frames}, "
                    f"translation range: {final_translations.min(axis=0)} to {final_translations.max(axis=0)}"
                )
            _animateTranslationKeyFrames(
                animLayer=animLayer,
                node=node,
                translations=final_translations,
                frameDuration=frameDuration,
            )
            root_translation_applied = True

    # If root translation was not applied, try to find root node by common names
    if not root_translation_applied:
        print("Warning: Root translation was not applied through normal mapping, trying fallback...")
        root_candidates = ["Pelvis", "pelvis", "Hips", "hips", "Root", "root", "mixamorig:Hips"]
        for candidate in root_candidates:
            if candidate in nodes_map:
                root_node = nodes_map[candidate]
                # Get initial translation for fallback node
                initial_trans = root_node.LclTranslation.Get()
                fallback_initial = np.array([initial_trans[0], initial_trans[1], initial_trans[2]])
                final_translations = translations + fallback_initial
                print(
                    f"Found root node by fallback: '{candidate}', initial_offset={fallback_initial}, applying translation..."
                )
                _animateTranslationKeyFrames(
                    animLayer=animLayer,
                    node=root_node,
                    translations=final_translations,
                    frameDuration=frameDuration,
                )
                root_translation_applied = True
                break

        if not root_translation_applied:
            print("ERROR: Could not find root node to apply translation!")
            print(f"Available nodes: {list(nodes_map.keys())}")

    return animStack


def _saveScene(filename, fbxManager, fbxScene, embed_textures=True):
    """Save the FBX scene to a file

    Args:
        filename: Output file path
        fbxManager: FBX manager instance
        fbxScene: FBX scene to save
        embed_textures: Whether to embed textures/media in the FBX file (default True)
    """
    # Configure IOSettings to embed textures/media
    ios = fbxManager.GetIOSettings()
    if embed_textures:
        ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
        ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)

    exporter = fbx.FbxExporter.Create(fbxManager, "")
    isInitialized = exporter.Initialize(filename, -1, ios)

    if isInitialized is False:
        raise Exception(f"Exporter failed to initialize. Error: {exporter.GetStatus().GetErrorString()}")

    exporter.Export(fbxScene)
    exporter.Destroy()


def _convert_smplh_to_woodfbx(
    template_fbx_path,
    npz_data,
    save_fn,
    fps=30,
    scale=100,
    smplh_to_fbx_mapping=None,
    clear_animations=True,
):
    """
    Convert SMPL-H parameters to FBX using a template FBX file.
    The template FBX skeleton is already consistent with SMPL-H, so we directly copy parameters.

    Args:
        template_fbx_path: Path to the template FBX file (e.g., boy_Rigging_smplx.fbx)
        npz_data: Dictionary containing SMPL-H parameters
                 - poses: (num_frames, 52, 3) or (num_frames, 156)
                 - trans: (num_frames, 3)
        save_fn: Output FBX file path
        fps: Frame rate
        scale: Scale factor for translation (default 100 for m to cm conversion)
        smplh_to_fbx_mapping: Custom mapping from SMPL-H joint names to FBX node names
        clear_animations: Whether to clear existing animations in the template

    Returns:
        bool: True if successful
    """
    # Prepare poses data
    poses = npz_data["poses"]
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()

    if len(poses.shape) == 2:
        # (num_frames, 156) -> (num_frames, 52, 3)
        poses = poses.reshape(poses.shape[0], -1, 3)

    # Convert axis-angle to rotation matrices: (num_frames, num_joints, 3, 3)
    rot_matrices = angle_axis_to_rotation_matrix(poses).numpy()

    # Prepare translation data
    trans = npz_data["trans"]
    if isinstance(trans, torch.Tensor):
        trans = trans.numpy()

    # Apply scale to translation
    translations = trans * scale

    # Create FBX manager and load template
    fbxManager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(fbxManager, fbx.IOSROOT)
    fbxManager.SetIOSettings(ios)

    print(f"Loading FBX template: {template_fbx_path}")
    fbxScene = _loadFbxScene(fbxManager, template_fbx_path)

    # Set time mode
    timeMode = fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
    fbxScene.GetGlobalSettings().SetTimeMode(timeMode)

    # Collect all nodes
    rootNode = fbxScene.GetRootNode()
    all_nodes = _collectAllNodes(rootNode)
    skeleton_nodes = _collectSkeletonNodes(rootNode)

    print(f"Found {len(all_nodes)} nodes in scene")
    print(f"Found {len(skeleton_nodes)} skeleton nodes: {list(skeleton_nodes.keys())}")

    # Use default mapping if not provided
    if smplh_to_fbx_mapping is None:
        smplh_to_fbx_mapping = _auto_detect_mapping(all_nodes)
        print(f"Auto-detected {len(smplh_to_fbx_mapping)} joint mappings")
        if "Pelvis" in smplh_to_fbx_mapping:
            print(f"  Root joint 'Pelvis' mapped to: '{smplh_to_fbx_mapping['Pelvis']}'")
        else:
            print(f"  WARNING: Root joint 'Pelvis' not found in mapping!")
            print(f"  Available nodes: {list(all_nodes.keys())[:20]}...")  # Show first 20 nodes

    # Clear existing animations if requested
    if clear_animations:
        _clearExistingAnimations(fbxScene)

    # Apply animation to skeleton
    _applyAnimationToSkeleton(
        fbxScene=fbxScene,
        nodes_map=all_nodes,
        rot_matrices=rot_matrices,
        translations=translations,
        fps=fps,
        smplh_to_fbx_mapping=smplh_to_fbx_mapping,
        name="SMPLH_Animation",
    )

    # Save to temporary file first, then copy to final destination
    os.makedirs(os.path.dirname(save_fn) if os.path.dirname(save_fn) else ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as tmp_f:
        temp_file = tmp_f.name

    try:
        _saveScene(temp_file, fbxManager, fbxScene)
        shutil.copy2(temp_file, save_fn)
        os.remove(temp_file)
        print(f"Successfully saved FBX to: {save_fn}")
    except Exception as e:
        print(f"Error saving FBX file: {e}")
        return False
    finally:
        fbxManager.Destroy()
        del fbxManager, fbxScene

    return os.path.exists(save_fn)


def _auto_detect_mapping(all_nodes):
    """Auto-detect the mapping from SMPL-H joints to FBX nodes"""
    mapping = {}
    for smplh_name in SMPLH_JOINT2NUM.keys():
        # Try exact match
        if smplh_name in all_nodes:
            mapping[smplh_name] = smplh_name
        # Try lowercase version
        elif SMPLH_TO_LOWERCASE_MAPPING.get(smplh_name) in all_nodes:
            mapping[smplh_name] = SMPLH_TO_LOWERCASE_MAPPING[smplh_name]
    return mapping


class SMPLH2WoodFBX:
    """
    Class to convert SMPL-H parameters to FBX using a template FBX file.
    The template FBX skeleton is already consistent with SMPL-H, so we directly copy parameters.
    No SMPL-H model assets (model.npz) required.

    Example usage:
        converter = SMPLH2WoodFBX(
            template_fbx_path="./assets/wooden_models/boy_Rigging_smplx.fbx"
        )

        # From npz file
        converter.convert_npz_to_fbx("motion.npz", "output.fbx", fps=30)

        # From parameters dict
        params = {
            "poses": poses_array,  # (num_frames, 52, 3) or (num_frames, 156)
            "trans": trans_array,  # (num_frames, 3)
        }
        converter.convert_params_to_fbx(params, "output.fbx")
    """

    def __init__(
        self,
        template_fbx_path: str = "./assets/wooden_models/boy_Rigging_smplx_tex.fbx",
        smplh_to_fbx_mapping: Optional[Dict[str, str]] = None,
        scale: float = 100,
    ):
        """
        Initialize the converter.

        Args:
            template_fbx_path: Path to the template FBX file
            smplh_to_fbx_mapping: Custom mapping from SMPL-H joint names to FBX node names
            scale: Scale factor for translation (default 100 for m to cm conversion)
        """
        print(f"[{self.__class__.__name__}] Template FBX: {template_fbx_path}")
        self.template_fbx_path = template_fbx_path
        self.smplh_to_fbx_mapping = smplh_to_fbx_mapping
        self.scale = scale

        # Analyze template FBX to detect joint names
        self._analyze_template()

    def _analyze_template(self):
        """Analyze the template FBX file to detect available skeleton nodes"""
        fbxManager = fbx.FbxManager.Create()
        ios = fbx.FbxIOSettings.Create(fbxManager, fbx.IOSROOT)
        fbxManager.SetIOSettings(ios)

        try:
            fbxScene = _loadFbxScene(fbxManager, self.template_fbx_path)
            rootNode = fbxScene.GetRootNode()

            self.all_template_nodes = list(_collectAllNodes(rootNode).keys())
            self.skeleton_template_nodes = list(_collectSkeletonNodes(rootNode).keys())

            print(f"[{self.__class__.__name__}] Template nodes: {len(self.all_template_nodes)}")
            print(f"[{self.__class__.__name__}] Skeleton nodes: {self.skeleton_template_nodes}")

            # Auto-detect mapping if not provided
            if self.smplh_to_fbx_mapping is None:
                self.smplh_to_fbx_mapping = self._auto_detect_mapping()
                print(f"[{self.__class__.__name__}] Auto-detected {len(self.smplh_to_fbx_mapping)} joint mappings")
        finally:
            fbxManager.Destroy()

    def _auto_detect_mapping(self):
        """Auto-detect the mapping from SMPL-H joints to FBX nodes"""
        mapping = {}
        for smplh_name in SMPLH_JOINT2NUM.keys():
            # Try exact match
            if smplh_name in self.all_template_nodes:
                mapping[smplh_name] = smplh_name
            # Try lowercase version
            elif SMPLH_TO_LOWERCASE_MAPPING.get(smplh_name) in self.all_template_nodes:
                mapping[smplh_name] = SMPLH_TO_LOWERCASE_MAPPING[smplh_name]
        return mapping

    def convert_npz_to_fbx(self, npz_file, outname, fps=30, clear_animations=True):
        """
        Convert an npz file containing SMPL-H parameters to FBX.

        Args:
            npz_file: Path to the npz file or dict containing SMPL-H parameters
            outname: Output FBX file path
            fps: Frame rate
            clear_animations: Whether to clear existing animations in template

        Returns:
            bool: True if successful
        """
        os.makedirs(os.path.dirname(outname) if os.path.dirname(outname) else ".", exist_ok=True)

        if isinstance(npz_file, str) and os.path.isfile(npz_file):
            npz_data = dict(np.load(npz_file, allow_pickle=True))
        else:
            npz_data = npz_file

        return _convert_smplh_to_woodfbx(
            template_fbx_path=self.template_fbx_path,
            npz_data=npz_data,
            save_fn=outname,
            fps=fps,
            scale=self.scale,
            smplh_to_fbx_mapping=self.smplh_to_fbx_mapping,
            clear_animations=clear_animations,
        )

    def convert_params_to_fbx(self, params, outname, clear_animations=True):
        """
        Convert SMPL-H parameters to FBX.

        Args:
            params: Dictionary containing SMPL-H parameters
                   - poses: (num_frames, 52, 3) or (num_frames, 156)
                   - trans: (num_frames, 3)
                   - mocap_framerate (optional): Frame rate
            outname: Output FBX file path
            clear_animations: Whether to clear existing animations in template

        Returns:
            bool: True if successful
        """
        fps = params.get("mocap_framerate", 30)
        os.makedirs(os.path.dirname(outname) if os.path.dirname(outname) else ".", exist_ok=True)

        npz_data = {
            "poses": params["poses"],
            "trans": params["trans"],
        }

        return _convert_smplh_to_woodfbx(
            template_fbx_path=self.template_fbx_path,
            npz_data=npz_data,
            save_fn=outname,
            fps=fps,
            scale=self.scale,
            smplh_to_fbx_mapping=self.smplh_to_fbx_mapping,
            clear_animations=clear_animations,
        )


if __name__ == "__main__":
    # python hymotion/utils/smplh2woodfbx.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    args = parser.parse_args()

    converter = SMPLH2WoodFBX(
        template_fbx_path="./assets/wooden_models/boy_Rigging_smplx_tex.fbx",
        scale=100,
    )

    if os.path.isdir(args.root):
        npzfiles = sorted(glob.glob(os.path.join(args.root, "*.npz")))
    else:
        if args.root.endswith(".npz"):
            npzfiles = [args.root]
        else:
            raise ValueError(f"Unknown file type: {args.root}")

    for npzfile in npzfiles:
        converter.convert_npz_to_fbx(npzfile, npzfile.replace(".npz", ".fbx").replace("motions", "motions_fbx"))
