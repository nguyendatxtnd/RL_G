#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

def main():
    rospy.init_node('odom_sub_pub', anonymous=True)

    # Khởi tạo subscriber và publisher
    subscriber = rospy.Subscriber("/husky_velocity_controller/odom", Odometry, odom_callback)
    publisher = rospy.Publisher("/odom", Odometry)

    rate = rospy.Rate(10)  # Tốc độ lặp là 10 Hz

    while not rospy.is_shutdown():
        # Kiểm tra xem có dữ liệu mới từ subscriber không
        if subscriber.get_num_connections() > 0:
            # Nhận message mới nhất từ subscriber
            try:
                odom_data = rospy.wait_for_message("/husky_velocity_controller/odom", Odometry)
         
                print(odom_data)
                publisher.publish(odom_data)
            except rospy.ROSException:
                pass

        rate.sleep()

def odom_callback(data):
    # Callback function cho subscriber
    # Trong trường hợp này, chúng ta không cần callback function, nhưng vẫn phải có nó
    pass

if __name__ == '__main__':
    main()
