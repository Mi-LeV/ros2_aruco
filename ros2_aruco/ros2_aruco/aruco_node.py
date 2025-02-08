import gz.transport13 as gz_transport
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from gz.msgs10.image_pb2 import Image
# Import necessary message types
from gz.msgs10 import image_pb2 as gz_msgs
from gz.msgs10 import pose_pb2 as gz_pose_msgs

class ArucoNode:
    def __init__(self):
        # Configuration parameters
        self.marker_size = 1.0  # Size of the markers in meters
        self.aruco_dictionary_id = "DICT_6X6_1000"
        self.image_topic = "/camera/image_raw"
        self.pose_topic = "/aruco_poses"

        self._img = None

        # Set up ArUco marker detection
        try:
            self.aruco_dictionary = cv2.aruco.Dictionary_get(
                getattr(cv2.aruco, self.aruco_dictionary_id)
            )
        except AttributeError:
            print(f"Invalid ArUco dictionary ID: {self.aruco_dictionary_id}")
            return

        self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        # Gazebo Transport components
        self.node = gz_transport.Node()

        # Subscribe to image topic with the correct message type
        self.node.subscribe(Image, self.image_topic, self.image_callback)

        # Use the correct message type for Pose (not Pose_V)
        self.pose_pub = self.node.advertise(self.pose_topic, gz_pose_msgs.Pose)
        print("Node initialized and ready.")

    def image_callback(self, msg):
        print("Image received")

        # Deserialize image message assuming it contains a raw buffer
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Check if the size matches the expected dimensions
        expected_size = msg.height * msg.width * 3
        if img_data.size != expected_size:
            print(f"Image size mismatch. Expected size: {expected_size}, but got: {img_data.size}", skip_first=True)
            return


        # If the image is RGB (3 channels), reshape accordingly
        # Uncomment the next line if your image is in RGB format
        self._img = img_data.reshape((msg.height, msg.width, 3))

        # Detect ArUco markers
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            self._img, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        if marker_ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, np.eye(3), np.zeros(5)
            )
            for i, marker_id in enumerate(marker_ids):
                pose = gz_pose_msgs.Pose()

                # Set position values
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                # Set orientation values (convert rotation vector to quaternion)
                rot_matrix = np.eye(3)
                rot_matrix, _ = cv2.Rodrigues(np.array(rvecs[i][0]))
                quat = R.from_matrix(rot_matrix).as_quat()

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                # Publish individual poses
                self.pose_pub.publish(pose)
            print("Published ArUco marker poses.")
        else:
            print("No ArUco markers detected.")
    
    def get_next_image(self, timeout=None):
        while self._img is None:
            pass
        ret_img = self._img
        self._img = None
        return ret_img



def main():
    node = ArucoNode()
    while True:
        img = node.get_next_image()
        cv2.imshow('pic-display', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
