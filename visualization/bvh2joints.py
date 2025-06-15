import numpy as np
import visualization.BVH_mod as BVH
import visualization.Animation as Animation
import visualization.joints2bvh as joints2bvh

class BVH2JointConvertor:
    def __init__(self):
        pass

    def convert(self, bvh_file_path):
        """
        Convert a BVH file to a numpy array of joint positions.

        :param bvh_file_path: Path to the .bvh file
        :return: joint positions of shape (N, 22, 3)
        """
        anim = BVH.load(bvh_file_path, need_quater=True)
        glb_positions = Animation.positions_global(anim)  # Shape: (frames, joints, 3)
        converter = joints2bvh.Joint2BVHConvertor()
        original_order_positions = glb_positions[: , converter.re_order_inv]
        return original_order_positions

if __name__ == "__main__":
    bvh_path = "generation/generation_name_nopredlen/animations/0/sample0_repeat0_len176.bvh"
    convertor = BVH2JointConvertor()
    joints = convertor.convert(bvh_path)
    print("Joints shape:", joints.shape)  # Expect (N_frames, 22, 3)
    np.save("extracted_joints.npy", joints)