##############################################################################################
#                                                                                            #
#  Filename: main.py                                                                         #
#  Author: htdv92                                                                            #
#  Date created: 22 / 02 / 2019                                                              #
#  Date Last-Modified: 13 / 03 / 2019                                                        #
#  Python Version: 3.7                                                                       #
#  Dependicies: csv, matplotlib, numpy, pylab(only for legend rengering)                     #
#  Description: Program for estimating the position of a VR headset given                    #
#               IMU data in a CSV format. Plots input data and positons                      #
#  Other Requirements: 'IMUData.csv' file in same directory                                  #
#                                                                                            #
##############################################################################################
#  Running the code: Running in the command line 'python main.py':                           #
#                    Creates input data plots                                                #
#                    Creates position plots as a single image (indivudal optiopn avaliable)  #
#                    Produces an animated plot and normal speed on repeat                   #
#                    Once closed, second animaton rendered at half speed                    #
##############################################################################################

import csv
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

# plt.rc('text', usetex=True)  # Use latex fonts for graph plotting, possible only if latex downloaded. Default: off
plt.rc('font', family='serif')


def quaternion_product(quat_a, quat_b):
    """
    Returns the product of two quaternions
    :param quat_a:
    :param quat_b:
    :return: quaternion_c = a * b, as a tuple
    """
    qua_c_0 = quat_a[0] * quat_b[0] - quat_a[1] * quat_b[1] - quat_a[2] * quat_b[2] - quat_a[3] * quat_b[3]
    qua_c_1 = quat_a[1] * quat_b[0] + quat_a[0] * quat_b[1] - quat_a[3] * quat_b[2] + quat_a[2] * quat_b[3]
    qua_c_2 = quat_a[2] * quat_b[0] + quat_a[3] * quat_b[1] + quat_a[0] * quat_b[2] - quat_a[1] * quat_b[3]
    qua_c_3 = quat_a[3] * quat_b[0] - quat_a[2] * quat_b[1] + quat_a[1] * quat_b[2] + quat_a[0] * quat_b[3]

    return qua_c_0, qua_c_1, qua_c_2, qua_c_3


def euler_to_quaternion(euler_angles):
    """
    Accepts XYZ angles in rad as a list and returns the equivalent quaternion
    :param euler_angles:
    :return: quaternion as tuple
    """
    cy = np.cos(euler_angles[2] * 0.5)  # yaw
    sy = np.sin(euler_angles[2] * 0.5)  # yaw
    cp = np.cos(euler_angles[1] * 0.5)  # pitch
    sp = np.sin(euler_angles[1] * 0.5)  # pitch
    cr = np.cos(euler_angles[0] * 0.5)  # roll
    sr = np.sin(euler_angles[0] * 0.5)  # roll

    quat_w = cy * cp * cr + sy * sp * sr  # w
    quat_x = cy * cp * sr - sy * sp * cr  # x
    quat_y = sy * cp * sr + cy * sp * cr  # t
    quat_z = sy * cp * cr - cy * sp * sr  # z

    return quat_w, quat_x, quat_y, quat_z


def quaternion_to_euler(qua):
    """
    Accepts 4 dimensional quaternion and returns 3 euler angles in rad as XYZ
    :param qua:
    :return: euler angles
    """
    # roll (x - axis rotation)
    sinr_cosp = 2.0 * (qua[0] * qua[1] + qua[2] * qua[3])  # w x y z
    cosr_cosp = 1.0 - 2.0 * (qua[1] * qua[1] + qua[2] * qua[2])  # x x y y
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y - axis rotation)

    sinp = 2.0 * (qua[0] * qua[2] - qua[3] * qua[1])  # w y z x
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z - axis rotation)
    siny_cosp = 2.0 * (qua[0] * qua[3] + qua[1] * qua[2])  # w z x y
    cosy_cosp = 1.0 - 2.0 * (qua[2] * qua[2] + qua[3] * qua[3])  # y y z z
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # euler_angle = (np.degrees(roll), np.degrees(pitch), np.degrees(yaw)) # convert to degress
    euler_angle = roll, pitch, yaw

    return euler_angle


def quaternion_conjugate(qua):
    """
    Returns the conjugate of a quaternion as a tuple
    :param qua:
    :return: quaternion conjugate
    """
    return qua[0], -qua[1], -qua[2], -qua[3]


def normalize_magnitude(vector):
    """
    Accepts a XYZ vector and returns the normalized version
    :param vector:
    :return: normalized vector
    """
    val_x = 0
    val_y = 0
    val_z = 0

    mag = np.linalg.norm(vector)  # finds the magnitude of the vector

    # check for NaN values of 0 for each vector
    if vector[0] != 0:
        val_x = vector[0] / mag
    if vector[1] != 0:
        val_y = vector[1] / mag
    if vector[2] != 0:
        val_z = vector[2] / mag

    return val_x, val_y, val_z


def convert_rotation_deg_rad(data):
    """
    Converts all values in the gyroscope from degrees to radians and returns the altered array
    :param data:
    :return:
    """
    data_rad = np.zeros(shape=(6959, 3))  # creates a new np.array for returning values

    for i in range(0, len(data)):  # loop through all gyroscope readings
        data_rad[i][0] = np.radians(data[i][0])
        data_rad[i][1] = np.radians(data[i][1])
        data_rad[i][2] = np.radians(data[i][2])

    return data_rad


def axis_angle_to_quaternion(axis, angle):
    """
    Used for finding the updated rotation using integration, accepts the tilt axis and angle
    :param axis:
    :param angle:
    :return: rotation quaternion
    """
    angle_2 = angle / 2
    quat_w = np.cos(angle_2)
    quat_x = np.sin(angle_2) * axis[0]
    quat_y = np.sin(angle_2) * axis[1]
    quat_z = np.sin(angle_2) * axis[2]

    return quat_w, quat_x, quat_y, quat_z


def normalize_data(data):
    """
    Normalizes all the values in an array
    :param data:
    :return: data_normalize
    """
    data_normalize = np.empty(shape=(6959, 3))

    for i in range(0, len(data)):  # iterate over both arrays
        data_normalize[i] = normalize_magnitude(data[i])  # normalize values and reallocate

    return data_normalize


def calculate_position(time, gyroscope):
    """
    Basic position integration calculation, dead reckoning filter, Problem 2
    :param time:
    :param gyroscope:
    :return: position estimation np array
    """
    position = np.zeros(shape=(6959, 4))
    position[0] = np.array([1, 0, 0, 0])

    for i in range(1, len(time)):  # iterate over each comparable time period
        change_theta = np.linalg.norm(gyroscope[i]) * (time[i] - time[i - 1])  # magnitude of gyroscope * change in time
        rotation = axis_angle_to_quaternion(normalize_magnitude(gyroscope[i]),
                                            change_theta)  # convert to rotation quaternion
        position[i] = quaternion_product(rotation,
                                         position[i - 1])  # find new position by rotating other postion by new rotation

    print("Estimated Position")
    return position


def calculate_tilt_correction(time, gyroscope, accelerometer, alpha_tilt):
    """
    Position integration that considers tilt correciton, Problem 3
    :param time:
    :param gyroscope:
    :param accelerometer:
    :param alpha_tilt:
    :return: position estimation np array
    """
    position_tilt = np.zeros(shape=(6959, 4))
    position_tilt[0] = np.array([1, 0, 0, 0])

    acc_average = [0, 0, 0]  # inital average
    for i in range(1, len(time)):
        # Inital Position Calculation, using gyroscope integration (see above) ----------------------------------------
        change_theta = np.linalg.norm(gyroscope[i]) * (time[i] - time[i - 1])
        rotation = axis_angle_to_quaternion(normalize_magnitude(gyroscope[i]), change_theta)
        position_tilt[i] = quaternion_product(rotation, position_tilt[i - 1])

        # Estimation of tilt error -----------------------------------------------------------------------------------
        acc = [0] + accelerometer[i].tolist()  # convert accelerometer to quaternion

        # convert to global frame = q*acc*q', convert back to 3 dimensional vector
        acc_p = np.array(quaternion_product(position_tilt[i],
                                            quaternion_product(acc, quaternion_conjugate(position_tilt[i])))[1:])

        acc_average = (acc_average * i + acc_p) / (i + 1)  # calculate running average for accelerometer

        # note, the up vector is the z-axis, (0,0,1), the dot product calculation below is therefore simplified
        phi = np.arccos(acc_average[2] / np.linalg.norm(acc_average))  # Tilt error, angle between acc_p and up vector
        v_tilt = (acc_average[1], -acc_average[0], 0)  # Tilt axis (y,-x,0)
        tilt_correction = axis_angle_to_quaternion(v_tilt, alpha_tilt * phi)  # overall tilt correction value

        position_tilt[i] = quaternion_product(tilt_correction, position_tilt[i])  # update position from tilt correction

    print("Estimated Drift Error")
    return position_tilt


def calculate_yaw_correction(time, gyroscope, accelerometer, magnetometer, alpha_tilt, alpha_yaw):
    """
    Position integration that considers tilt correciton and yaw drift (complementary filter), Problem 4
    :param time:
    :param gyroscope:
    :param accelerometer:
    :param magnetometer:
    :param alpha_tilt:
    :param alpha_yaw:
    :return: position estimation np array
    """
    position_tilt_yaw = np.zeros(shape=(6959, 4))
    position_tilt_yaw[0] = np.array([1, 0, 0, 0])

    inital_qua = [1, 0, 0, 0]
    inital_mag_global = quaternion_product(inital_qua, quaternion_product([0] + magnetometer[0].tolist(),
                                                                          quaternion_conjugate(inital_qua)))[1:]
    inital_theta = np.arctan2(inital_mag_global[1], inital_mag_global[0])

    acc_average = [0, 0, 0]
    # mag_average = [0, 0, 0]
    # averaging for the magnetometer has been removed since better results were found without it

    for i in range(1, len(time)):
        # Inital Position Calculation, using gyroscope integration (see above) ----------------------------------------
        change_theta = np.linalg.norm(gyroscope[i]) * (time[i] - time[i - 1])
        rotation = axis_angle_to_quaternion(normalize_magnitude(gyroscope[i]), change_theta)
        position_tilt_yaw[i] = quaternion_product(rotation, position_tilt_yaw[i - 1])

        # Estimation of tilt error (see above) ------------------------------------------------------------------------
        acc = [0] + accelerometer[i].tolist()
        acc_p = np.array(quaternion_product(position_tilt_yaw[i], quaternion_product(acc, quaternion_conjugate(
            position_tilt_yaw[i])))[1:])
        acc_average = (acc_average * i + acc_p) / (i + 1)
        phi = np.arccos(acc_average[2] / np.linalg.norm(acc_average))
        v_tilt = (acc_average[1], -acc_average[0], 0)
        tilt_correction = axis_angle_to_quaternion(v_tilt, alpha_tilt * phi)

        position_tilt_yaw[i] = quaternion_product(tilt_correction, position_tilt_yaw[i])

        # Estimation of yaw drift -----------------------------------------------------------------------------------
        mag = [0] + magnetometer[i].tolist()  # 4 dimensional quaterninon for magnetometer
        # transform to global frame, convert back to 3 dimensional vector
        mag_p = np.array(quaternion_product(position_tilt_yaw[i],
                                            quaternion_product(mag, quaternion_conjugate(position_tilt_yaw[i])))[1:])
        # mag_average = (mag_average * i + mag_p) / (i + 1)

        theta = np.arctan2(mag_p[1], mag_p[0])  # Angular difference, X and Y

        # yaw drift found by comparing to inital reading
        yaw_drift = axis_angle_to_quaternion((0, 0, 1),
                                             (alpha_yaw * (theta - inital_theta)))
        position_tilt_yaw[i] = np.array(quaternion_product(yaw_drift, position_tilt_yaw[i]))

    print("Estimated Yaw Error")
    return position_tilt_yaw


def plot_data(time, y_data, y_label, title):
    """
    Used for plotting the intial data, accepts the time np.array, the data to be plotted and identification
    :param time:
    :param y_data:
    :param y_label:
    :param title:
    :return: saves the fig with the appropriate title
    """
    plt.figure()

    font = {'family': 'serif',
            'weight': 'bold',
            # 'size': 18
            }
    plt.rc('font', **font)  # Set the font so it is consitent, not needed if in latex mode

    x_data, y_data, z_data = zip(*y_data)  # zips up data for each coordiante for individual plotting

    data_label_and_colour = ((x_data, 'blue', 'x'), (y_data, 'green', 'y'), (z_data, 'red', 'z'))  # tuple in format

    for data, color, label in data_label_and_colour:
        plt.plot(time, data, color=color, label=label, linewidth=0.5)

    plt.xlabel("Time (seconds)")  # label for x-axis
    plt.ylabel(y_label)  # label for y-axis

    plt.legend()  # plot legend
    axes = plt.gca()

    # formatting for axis
    axes.spines['left'].set_color('black')
    axes.spines['bottom'].set_color('black')
    axes.xaxis.label.set_color('black')
    axes.yaxis.label.set_color('black')
    axes.tick_params(axis='x', colors='black')
    axes.tick_params(axis='y', colors='black')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plt.savefig(title + ".png")  # saves the figure with the title name e.g. accelerometer
    print(title + " plotted")  # notify the console
    plt.close()


def plot_position(time, position_data, y_label, title):
    """
    Plot the calculated data individually. Formating same as data above.
    Converts data to deg angles from rad and saves the result
    :param time:
    :param position_data:
    :param y_label:
    :param title:
    :return:
    """
    plt.figure()
    font = {'family': 'serif',
            'weight': 'bold',
            # 'size': 18
            }
    plt.rc('font', **font)

    euler_position = np.zeros(shape=(6959, 3))  # conversion to euler angles
    for i in range(0, len(time)):
        euler_position[i] = np.degrees(quaternion_to_euler(position_data[i]))  # convert to degress before plotting

    x_data, y_data, z_data = zip(*euler_position)
    data_label_and_colour = (
        (x_data, 'blue', 'x'),
        (y_data, 'green', 'y'),
        (z_data, 'red', 'z'),
    )

    for data, color, label in data_label_and_colour:
        plt.plot(time, data, color=color, label=label, linewidth=1)

    plt.xlabel("Time (seconds)")
    plt.ylabel(y_label)

    plt.legend()
    axes = plt.gca()
    axes.spines['left'].set_color('black')
    axes.spines['bottom'].set_color('black')
    axes.xaxis.label.set_color('black')
    axes.yaxis.label.set_color('black')
    axes.tick_params(axis='x', colors='black')
    axes.tick_params(axis='y', colors='black')
    # axes.spines['top'].set_visible(False)
    # axes.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(title + ".png")
    print(title + " plotted")
    plt.close()


def plot_all_positions(time, position_data, position_data_tilt, position_data_yaw):
    """
    Plots all position data in a single figure. Therefore only requries one legend and axis labeling, neater output
    :param time:
    :param position_data:
    :param position_data_tilt:
    :param position_data_yaw:
    :return:
    """
    fig = plt.figure(figsize=(12, 4))

    font = {'family': 'serif',
            'weight': 'bold',
            # 'size': 12
            }
    plt.rc('font', **font)

    def convert_data_to_euler(pos):
        euler_pos = np.zeros(shape=(6959, 3))
        for i in range(0, len(time)):
            euler_pos[i] = np.degrees(quaternion_to_euler(pos[i]))  # convert to degrees before plotting

        x_data, y_data, z_data = zip(*euler_pos)
        data_label_and_colour = (
            (x_data, 'blue', 'x'),
            (y_data, 'green', 'y'),
            (z_data, 'red', 'z'),
        )

        return data_label_and_colour

    ax = fig.add_subplot(131)
    ax.set_title("Position")
    ax.set_ylabel("Tri-axial Euler angles $(deg)$")
    ax.set_xlabel("Time $(Seconds)$")

    ax_tilt = fig.add_subplot(132)
    ax_tilt.set_title("Position with Tilt Correction")
    ax_tilt.set_xlabel("Time $(Seconds)$")
    ax_tilt.set_yticklabels([])

    ax_yaw = fig.add_subplot(133)
    ax_yaw.set_title("Position with Tilt and Yaw Correction")
    ax_yaw.set_xlabel("Time $(Seconds)$")
    ax_yaw.set_yticklabels([])

    # Plot data using the same technique as in single position plotting
    for data, color, label in convert_data_to_euler(position_data):
        ax.plot(time, data, color=color, label=label, linewidth=1)
    for data, color, label in convert_data_to_euler(position_data_tilt):
        ax_tilt.plot(time, data, color=color, label=label, linewidth=1)
    for data, color, label in convert_data_to_euler(position_data_yaw):
        ax_yaw.plot(time, data, color=color, label=label, linewidth=1)

    plt.legend()
    plt.tight_layout()
    plt.savefig("positions.png")  # transparent=True)
    print("Position plotted")
    plt.close()


def animated_plots(time, position, position_tilt, position_tilt_yaw, speed):
    """
    Function that accepts all the input data and a desired speed and renders an animated plot of headset movement
    :param time:
    :param position:
    :param position_tilt:
    :param position_tilt_yaw:
    :param speed: e.g. 1, 0.5, 5
    :return:
    """
    fig = plt.figure(figsize=(15, 4))
    time_per_frame = 3.906295  # in milliseconds

    # The number of frames for aniamtion rendering to skip (automatically calculated from speed), however slower
    # computers will need to alter the value manually for smooth rendering
    frame_skips = int(speed * 10)
    # frame_skips = 50
    interval = (frame_skips * time_per_frame) / speed  # interval between frames in final rendering
    colors = ["blue", "green", "red"]
    font = {'family': 'serif',
            'weight': 'bold',
            # 'size': 10
            }
    plt.rc('font', **font)

    ax = fig.add_subplot(131, projection='3d')  # 337 3x3, position 7
    ax_tilt = fig.add_subplot(132, projection='3d')
    ax_yaw = fig.add_subplot(133, projection='3d')

    ax.set_xlabel('X', labelpad=-12)
    ax.set_ylabel('Y', labelpad=-12)
    ax.set_zlabel('Z', labelpad=-12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("Position")

    ax_tilt.set_xlabel('X', labelpad=-12)
    ax_tilt.set_ylabel('Y', labelpad=-12)
    ax_tilt.set_zlabel('Z', labelpad=-12)
    ax_tilt.set_xlim(-1, 1)
    ax_tilt.set_ylim(-1, 1)
    ax_tilt.set_zlim(-1, 1)
    ax_tilt.set_xticklabels([])
    ax_tilt.set_yticklabels([])
    ax_tilt.set_zticklabels([])
    ax_tilt.set_title("Position with Tilt Correction")

    ax_yaw.set_xlabel('X', labelpad=-12)
    ax_yaw.set_ylabel('Y', labelpad=-12)
    ax_yaw.set_zlabel('Z', labelpad=-12)
    ax_yaw.set_xlim(-1, 1)
    ax_yaw.set_ylim(-1, 1)
    ax_yaw.set_zlim(-1, 1)
    ax_yaw.set_xticklabels([])
    ax_yaw.set_yticklabels([])
    ax_yaw.set_zticklabels([])
    ax_yaw.set_title("Position with Tilt and Yaw Correction")

    # title = ax.text(0.5, 0.85, "","",bbox=
    # {'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")

    origin = np.zeros(3)  # origin vector point
    x, y, z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # inital 3 perpendicular vectors

    # Draw Axis References, draw the inital axis orientation on each plot
    ax.plot((1, -1), (0, 0), (0, 0), color="grey")
    ax.plot((0, 0), (1, -1), (0, 0), color="grey")
    ax.plot((0, 0), (0, 0), (1, -1), color="grey")

    ax_tilt.plot((1, -1), (0, 0), (0, 0), color="grey")
    ax_tilt.plot((0, 0), (1, -1), (0, 0), color="grey")
    ax_tilt.plot((0, 0), (0, 0), (1, -1), color="grey")

    ax_yaw.plot((1, -1), (0, 0), (0, 0), color="grey")
    ax_yaw.plot((0, 0), (1, -1), (0, 0), color="grey")
    ax_yaw.plot((0, 0), (0, 0), (1, -1), color="grey")

    # quiver represents the 3 3-dimensional perpendicular vectors, plotted
    quiver_position = ax.quiver(origin, origin, origin, x, y, z, color=colors, arrow_length_ratio=0.1)
    quiver_tilt = ax_tilt.quiver(origin, origin, origin, x, y, z, color=colors, arrow_length_ratio=0.1)
    quiver_yaw = ax_yaw.quiver(origin, origin, origin, x, y, z, color=colors, arrow_length_ratio=0.1)

    x, y, z = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def update(i):  # updates the current frame
        # Alter Timer Count
        val = round(i * 0.003906295, 3)  # used for showing the stopwatch at the top of the figure
        fig.suptitle(
            "Time: " + str(val) + "s" + "                  " + "  Speed:" + "" + str(
                speed) + "x", fontsize=20)

        # Update position calculations for each subplot,  q*vector*q'
        pos = position[i].tolist()
        pos_tilt = position_tilt[i].tolist()
        pos_tilt_yaw = position_tilt_yaw[i].tolist()

        x_normal = quaternion_product(pos, quaternion_product(x.tolist(), quaternion_conjugate(pos)))[1:]
        y_normal = quaternion_product(pos, quaternion_product(y.tolist(), quaternion_conjugate(pos)))[1:]
        z_normal = quaternion_product(pos, quaternion_product(z.tolist(), quaternion_conjugate(pos)))[1:]

        x_tilt = quaternion_product(pos_tilt, quaternion_product(x.tolist(), quaternion_conjugate(pos)))[1:]
        y_tilt = quaternion_product(pos_tilt, quaternion_product(y.tolist(), quaternion_conjugate(pos)))[1:]
        z_tilt = quaternion_product(pos_tilt, quaternion_product(z.tolist(), quaternion_conjugate(pos)))[1:]

        x_yaw = quaternion_product(pos_tilt_yaw, quaternion_product(x.tolist(), quaternion_conjugate(pos)))[1:]
        y_yaw = quaternion_product(pos_tilt_yaw, quaternion_product(y.tolist(), quaternion_conjugate(pos)))[1:]
        z_yaw = quaternion_product(pos_tilt_yaw, quaternion_product(z.tolist(), quaternion_conjugate(pos)))[1:]

        quiver_position.set_segments([[origin, x_normal], [origin, y_normal], [origin, z_normal]])
        quiver_tilt.set_segments([[origin, x_tilt], [origin, y_tilt], [origin, z_tilt]])
        quiver_yaw.set_segments([[origin, x_yaw], [origin, y_yaw], [origin, z_yaw]])

        return quiver_position, quiver_tilt, quiver_yaw,

    ani = animation.FuncAnimation(fig, update, range(0, len(time), frame_skips), interval=interval, repeat=True)

    # blit=False
    # plt.tight_layout()
    plt.show()
    plt.close()
    print("Animated Plots Rendered with speed: " + str(speed) + "x")


def read_data(filename):
    time = []
    gyroscope = []
    accelerometer = []
    magnetometer = []

    with open(filename) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        next(read_csv)

        for row in read_csv:
            time.append(float(row[0]))
            gyroscope.append(np.array([float(row[1]), float(row[2]), float(row[3])]))
            accelerometer.append(np.array([float(row[4]), float(row[5]), float(row[6])]))
            magnetometer.append(np.array([float(row[7]), float(row[8]), float(row[9])]))

    print("Data Read")  # notify console
    # return lists as np.arrays for accessing speed, but build them as normal lists for speed creating
    return np.array(time), np.array(gyroscope), np.array(accelerometer), np.array(magnetometer)


def main():
    time, gyroscope, accelerometer, magnetometer = read_data('IMUData.csv')  # reads data into 4 np.arrays

    # Data Plotting
    plot_data(time, gyroscope, "Tri-axial Angular Rate in $deg/s$", "gyroscope")
    plot_data(time, accelerometer, "Tri-Axial Acceleration in $g (m/s^2)$", "acceleration")
    plot_data(time, magnetometer, "Tri-Axial Magnetic flux in $Gauss (G)$", "magnetometer")

    # Format Data
    gyroscope = convert_rotation_deg_rad(gyroscope)  # converts values deg- rad and reallocates it to the array
    accelerometer = normalize_data(accelerometer)  # normalizes values
    magnetometer = normalize_data(magnetometer)  # normalizes values

    # Position Calculations
    alpha_tilt = 0.05531  # Optimal alpha values allocated
    alpha_yaw = 0.00001272
    position = calculate_position(time, gyroscope)  # Position valeus using gyroscope integration
    position_tilt = calculate_tilt_correction(time, gyroscope, accelerometer,
                                              alpha_tilt)  # tilt correction incorporated
    position_tilt_yaw = calculate_yaw_correction(time, gyroscope, accelerometer, magnetometer, alpha_tilt,
                                                 alpha_yaw)  # tilt and yaw correction incorporated

    # Print results for last value
    # pos = np.degrees(quaternion_to_euler(position[len(time) - 1]))
    # tilt = np.degrees(quaternion_to_euler(position_tilt[len(time) - 1]))
    # yaw = np.degrees(quaternion_to_euler(position_tilt_yaw[len(time) - 1]))
    # print(pos, distance_to_zero(pos))
    # print(tilt, distance_to_zero(tilt))
    # print(yaw, distance_to_zero(yaw))

    # Position Plotting, individual and as a group
    plot_position(time, position, "Tri-axial Euler angles in deg", "position")
    plot_position(time, position_tilt, "Tri-axial Euler angles in deg", "positionTilt")
    plot_position(time, position_tilt_yaw, "Tri-axial Euler angles in deg", "positionTiltYaw")
    plot_all_positions(time, position, position_tilt, position_tilt_yaw)  # Plots all position estimations in one figure

    # Animated Ploting
    animated_plots(time, position, position_tilt, position_tilt_yaw, 1)  # last parameter is speed
    animated_plots(time, position, position_tilt, position_tilt_yaw, 0.5)  # last parameter is speed


# ---------------------------------------------------------------------------------------------------------------------
# Additional functions used during testing, searching for alpha values and experimentation

def testing():
    """
    Used for testing new functions, such as searching for appropriate alpha values
    """
    time, gyroscope, accelerometer, magnetometer = read_data('IMUData.csv')
    gyroscope = convert_rotation_deg_rad(gyroscope)
    accelerometer, magnetometer = normalize_data(accelerometer)  # normalizes values
    magnetometer = normalize_data(magnetometer)  # normalizes values

    print(binary_search_alpha(time, gyroscope, accelerometer, magnetometer))


def binary_search_alpha(time, gyroscope, accelerometer, magnetometer):
    """
    A Simple bianry search for investigation alpha values between a certain range
    :param time:
    :param gyroscope:
    :param accelerometer:
    :param magnetometer:
    :return:
    """
    alpha_lower = 0.00001
    alpha_upper = 0.0000128

    for i in range(0, 20):
        upper_value = distance_to_zero(np.degrees(quaternion_to_euler(
            calculate_yaw_correction(time, gyroscope, accelerometer, magnetometer, 0.05531, alpha_upper)[
                len(time) - 1])))

        lower_value = distance_to_zero(
            np.degrees(quaternion_to_euler(
                calculate_yaw_correction(time, gyroscope, accelerometer, magnetometer, 0.05531, alpha_lower)[
                    len(time) - 1])))

        print(
            str(i) + " " + str(lower_value) + " " + str(upper_value) + " " + str(alpha_lower) + " " + str(alpha_upper))

        if lower_value < upper_value:
            alpha_upper = ((alpha_upper - alpha_lower) / 2) + alpha_lower
        else:
            alpha_lower = alpha_upper - ((alpha_upper - alpha_lower) / 2)

    return (alpha_upper - alpha_lower) / 2 + alpha_lower


def distance_to_zero(angle):
    """
    A representation of how close the set of angles are to zero
    :param angle:
    :return:
    """
    return abs(angle[0]) + abs(angle[1]) + abs(angle[2])


def plot_legend():
    """
    Function for creatign a single legend seperate to the other figures, saves repeatign the smae legend
    :return:
    """
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(3, 2))
    ax = fig.add_subplot(333)
    lines = ax.plot(range(20), 'blue', range(20), 'green', range(20), 'red')
    figlegend.legend(lines, ('X', 'Y', 'Z'), 'center')
    figlegend.savefig('legend.png')


def table_alpha_results():
    """
    Prints to console results in latex table format, for ease of referencing and recording results
    :return:
    """
    time, gyroscope, accelerometer, magnetometer = read_data('IMUData.csv')  # reads data into 4 np.arrays

    # Format Data
    gyroscope = convert_rotation_deg_rad(gyroscope)  # converts values deg- rad and reallocates it to the array
    accelerometer = normalize_data(accelerometer)  # normalizes values
    magnetometer = normalize_data(magnetometer)  # normalizes values

    # Position Calculations
    def calculate(alpha_tilt, alpha_yaw):
        position = calculate_yaw_correction(time, gyroscope, accelerometer, magnetometer, alpha_tilt, alpha_yaw)
        pos = np.degrees(quaternion_to_euler(position[len(time) - 1]))

        if alpha_yaw == 0:
            print(
                "     " + str(alpha_tilt) + " & " + str(round(pos[0], 3)) + " & " + str(round(pos[1], 3)) + " & " + str(
                    round(pos[2], 3)) + " & " + str(round(distance_to_zero(pos), 3)) + " \\\ \hline ")
        else:
            print("     " + str(alpha_yaw) + " & " + str(round(pos[0], 3)) + " & " + str(round(pos[1], 3)) + " & " +
                  str(round(pos[2], 3)) + " & " + str(round(distance_to_zero(pos), 3)) + " \\\ \hline ")

    calculate(0, 0)
    calculate(1, 0)
    calculate(0.1, 0)
    calculate(0.01, 0)
    calculate(0.001, 0)
    calculate(0.0001, 0)
    calculate(0.00001, 0)
    calculate(0.05, 0)
    calculate(0.05531, 0)

    print()

    calculate(0.05531, 1)
    calculate(0.05531, 0.1)
    calculate(0.05531, 0.01)
    calculate(0.05531, 0.001)
    calculate(0.05531, 0.0001)
    calculate(0.05531, 0.00001)
    calculate(0.05531, 0.000001)
    calculate(0.05531, 0.000012)
    calculate(0.05531, 0.000012724)


main()
