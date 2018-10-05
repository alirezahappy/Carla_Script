#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import os
import numpy as np

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.transform import Transform #, Translation, Rotation, Scale
from carla import image_converter
import cv2


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 15
    frames_per_episode = 2030
    #              [0  , 1  , 2  , 3  , 4  , 5  , 6 , 7, 8  , 9  , 10, 11, 12, 13, 14]
    vehicles_num = [60, 60, 70, 50, 60, 60, 80, 60, 60, 60, 50, 70, 60, 50, 50] 

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=False,
                    NumberOfVehicles= vehicles_num[episode],#random.choice([0, 20, 15, 20, 25, 21, 24, 18, 40, 35, 25, 30]), #25,
                    NumberOfPedestrians=50,
                    DisableTwoWheeledVehicles=False,
                    WeatherId= episode, #1, #random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.


                #### FRONT STEREO ####

                # LEFT RGB CAMERA
                camera_l = Camera('LeftCameraRGB', PostProcessing='SceneFinal')
                camera_l.set_image_size(800, 600)
                camera_l.set_position(1.30, -0.27, 1.50)
                settings.add_sensor(camera_l)

                # LEFT DEPTH
                camera_ld = Camera('LeftCameraDepth', PostProcessing='Depth')
                camera_ld.set_image_size(800, 600)
                camera_ld.set_position(1.30, -0.27, 1.50)
                settings.add_sensor(camera_ld)

                # LEFT SEGMENTATION
                camera_ls = Camera('LeftCameraSeg', PostProcessing='SemanticSegmentation')
                camera_ls.set_image_size(800, 600)
                camera_ls.set_position(1.30, -0.27, 1.50)
                settings.add_sensor(camera_ls)

                # RIGHT RGB CAMERA
                camera_r = Camera('RightCameraRGB', PostProcessing='SceneFinal')
                camera_r.set_image_size(800, 600)
                camera_r.set_position(1.30, 0.27, 1.50)
                settings.add_sensor(camera_r)

                # RIGHT DEPTH
                camera_rd = Camera('RightCameraDepth', PostProcessing='Depth')
                camera_rd.set_image_size(800, 600)
                camera_rd.set_position(1.30, 0.27, 1.50)
                settings.add_sensor(camera_rd)

                # RIGHT SEGMENTATION
                camera_rs = Camera('RightCameraSeg', PostProcessing='SemanticSegmentation')
                camera_rs.set_image_size(800, 600)
                camera_rs.set_position(1.30, 0.27, 1.50)
                settings.add_sensor(camera_rs)


                #### -45 DEGREE STEREO CAMERA ####

                # LEFT STEREO -45 DEGREE RGB
                camera_45_n_l = Camera('45_N_LeftCameraRGB', PostProcessing='SceneFinal')
                camera_45_n_l.set_image_size(800, 600)
                camera_45_n_l.set_position(0.8, -1.0, 1.50) # [X, Y, Z]
                camera_45_n_l.set_rotation(0, -45.0, 0)     # [pitch(Y), yaw(Z), roll(X)]
                settings.add_sensor(camera_45_n_l)

                # RIGHT STEREO -45 DEGREE RGB
                camera_45_n_r = Camera('45_N_RightCameraRGB', PostProcessing='SceneFinal')
                camera_45_n_r.set_image_size(800, 600)
                camera_45_n_r.set_position(1.2, -0.6, 1.50)
                camera_45_n_r.set_rotation(0, -45.0, 0)
                settings.add_sensor(camera_45_n_r)

                # LEFT STEREO -45 DEGREE DEPTH
                camera_45_n_ld = Camera('45_N_LeftCameraDepth', PostProcessing='Depth')
                camera_45_n_ld.set_image_size(800, 600)
                camera_45_n_ld.set_position(0.8, -1.0, 1.50)
                camera_45_n_ld.set_rotation(0, -45.0, 0)
                settings.add_sensor(camera_45_n_ld)

                # RIGHT STEREO -45 DEGREE DEPTH
                camera_45_n_rd = Camera('45_N_RightCameraDepth', PostProcessing='Depth')
                camera_45_n_rd.set_image_size(800, 600)
                camera_45_n_rd.set_position(1.2, -0.6, 1.50)
                camera_45_n_rd.set_rotation(0, -45.0, 0)
                settings.add_sensor(camera_45_n_rd)

                # LEFT STEREO -45 DEGREE SEGMENTATION
                camera_45_n_ls = Camera('45_N_LeftCameraSeg', PostProcessing='SemanticSegmentation')
                camera_45_n_ls.set_image_size(800, 600)
                camera_45_n_ls.set_position(0.8, -1.0, 1.50)
                camera_45_n_ls.set_rotation(0, -45.0, 0)
                settings.add_sensor(camera_45_n_ls)

                # RIGHT STEREO -45 DEGREE SEGMENTATION
                camera_45_n_rs = Camera('45_N_RightCameraSeg', PostProcessing='SemanticSegmentation')
                camera_45_n_rs.set_image_size(800, 600)
                camera_45_n_rs.set_position(1.2, -0.6, 1.50)
                camera_45_n_rs.set_rotation(0, -45.0, 0)
                settings.add_sensor(camera_45_n_rs)

                
                #### +45 DEGREE STEREO CAMERA ####

                # LEFT STEREO +45 DEGREE RGB
                camera_45_p_l = Camera('45_P_LeftCameraRGB', PostProcessing='SceneFinal')
                camera_45_p_l.set_image_size(800, 600)
                camera_45_p_l.set_position(1.2, 0.6, 1.50) # [X, Y, Z]
                camera_45_p_l.set_rotation(0, 45.0, 0)     # [pitch(Y), yaw(Z), roll(X)]
                settings.add_sensor(camera_45_p_l)

                # RIGHT STEREO +45 DEGREE RGB
                camera_45_p_r = Camera('45_P_RightCameraRGB', PostProcessing='SceneFinal')
                camera_45_p_r.set_image_size(800, 600)
                camera_45_p_r.set_position(0.8, 1.0, 1.50)
                camera_45_p_r.set_rotation(0, 45.0, 0)
                settings.add_sensor(camera_45_p_r)

                # LEFT STEREO +45 DEGREE DEPTH
                camera_45_p_ld = Camera('45_P_LeftCameraDepth', PostProcessing='Depth')
                camera_45_p_ld.set_image_size(800, 600)
                camera_45_p_ld.set_position(1.2, 0.6, 1.50)
                camera_45_p_ld.set_rotation(0, 45.0, 0)
                settings.add_sensor(camera_45_p_ld)

                # RIGHT STEREO +45 DEGREE DEPTH
                camera_45_p_rd = Camera('45_P_RightCameraDepth', PostProcessing='Depth')
                camera_45_p_rd.set_image_size(800, 600)
                camera_45_p_rd.set_position(0.8, 1.0, 1.50)
                camera_45_p_rd.set_rotation(0, 45.0, 0)
                settings.add_sensor(camera_45_p_rd)

                # LEFT STEREO +45 DEGREE SEGMENTATION
                camera_45_p_ls = Camera('45_P_LeftCameraSeg', PostProcessing='SemanticSegmentation')
                camera_45_p_ls.set_image_size(800, 600)
                camera_45_p_ls.set_position(1.2, 0.6, 1.50)
                camera_45_p_ls.set_rotation(0, 45.0, 0)
                settings.add_sensor(camera_45_p_ls)

                # RIGHT STEREO +45 DEGREE SEGMENTATION
                camera_45_p_rs = Camera('45_P_RightCameraSeg', PostProcessing='SemanticSegmentation')
                camera_45_p_rs.set_image_size(800, 600)
                camera_45_p_rs.set_position(0.8, 1.0, 1.50)
                camera_45_p_rs.set_rotation(0, 45.0, 0)
                settings.add_sensor(camera_45_p_rs)

                
                #### -90 DEGREE STEREO CAMERA ####

                # LEFT STEREO -90 DEGREE RGB
                camera_90_n_l = Camera('90_N_LeftCameraRGB', PostProcessing='SceneFinal')
                camera_90_n_l.set_image_size(800, 600)
                camera_90_n_l.set_position(-0.27, -1.0, 1.50) # [X, Y, Z]
                camera_90_n_l.set_rotation(0, -90.0, 0)     # [pitch(Y), yaw(Z), roll(X)]
                settings.add_sensor(camera_90_n_l)

                # RIGHT STEREO -90 DEGREE RGB
                camera_90_n_r = Camera('90_N_RightCameraRGB', PostProcessing='SceneFinal')
                camera_90_n_r.set_image_size(800, 600)
                camera_90_n_r.set_position(0.27, -1.0, 1.50)
                camera_90_n_r.set_rotation(0, -90.0, 0)
                settings.add_sensor(camera_90_n_r)

                # LEFT STEREO -90 DEGREE DEPTH
                camera_90_n_ld = Camera('90_N_LeftCameraDepth', PostProcessing='Depth')
                camera_90_n_ld.set_image_size(800, 600)
                camera_90_n_ld.set_position(-0.27, -1.0, 1.50)
                camera_90_n_ld.set_rotation(0, -90.0, 0)
                settings.add_sensor(camera_90_n_ld)

                # RIGHT STEREO -90 DEGREE DEPTH
                camera_90_n_rd = Camera('90_N_RightCameraDepth', PostProcessing='Depth')
                camera_90_n_rd.set_image_size(800, 600)
                camera_90_n_rd.set_position(0.27, -1.0, 1.50)
                camera_90_n_rd.set_rotation(0, -90.0, 0)
                settings.add_sensor(camera_90_n_rd)

                # LEFT STEREO -90 DEGREE SEGMENTATION
                camera_90_n_ls = Camera('90_N_LeftCameraSeg', PostProcessing='SemanticSegmentation')
                camera_90_n_ls.set_image_size(800, 600)
                camera_90_n_ls.set_position(-0.27, -1.0, 1.50)
                camera_90_n_ls.set_rotation(0, -90.0, 0)
                settings.add_sensor(camera_90_n_ls)

                # RIGHT STEREO -90 DEGREE SEGMENTATION
                camera_90_n_rs = Camera('90_N_RightCameraSeg', PostProcessing='SemanticSegmentation')
                camera_90_n_rs.set_image_size(800, 600)
                camera_90_n_rs.set_position(0.27, -1.0, 1.50)
                camera_90_n_rs.set_rotation(0, -90.0, 0)
                settings.add_sensor(camera_90_n_rs)


                #### +90 DEGREE STEREO CAMERA ####

                # LEFT STEREO +90 DEGREE RGB
                camera_90_p_l = Camera('90_P_LeftCameraRGB', PostProcessing='SceneFinal')
                camera_90_p_l.set_image_size(800, 600)
                camera_90_p_l.set_position(0.27, 1.0, 1.50) # [X, Y, Z]
                camera_90_p_l.set_rotation(0, 90.0, 0)     # [pitch(Y), yaw(Z), roll(X)]
                settings.add_sensor(camera_90_p_l)

                # RIGHT STEREO +90 DEGREE RGB
                camera_90_p_r = Camera('90_P_RightCameraRGB', PostProcessing='SceneFinal')
                camera_90_p_r.set_image_size(800, 600)
                camera_90_p_r.set_position(-0.27, 1.0, 1.50)
                camera_90_p_r.set_rotation(0, 90.0, 0)
                settings.add_sensor(camera_90_p_r)

                # LEFT STEREO +90 DEGREE DEPTH
                camera_90_p_ld = Camera('90_P_LeftCameraDepth', PostProcessing='Depth')
                camera_90_p_ld.set_image_size(800, 600)
                camera_90_p_ld.set_position(0.27, 1.0, 1.50)
                camera_90_p_ld.set_rotation(0, 90.0, 0)
                settings.add_sensor(camera_90_p_ld)

                # RIGHT STEREO +90 DEGREE DEPTH
                camera_90_p_rd = Camera('90_P_RightCameraDepth', PostProcessing='Depth')
                camera_90_p_rd.set_image_size(800, 600)
                camera_90_p_rd.set_position(-0.27, 1.0, 1.50)
                camera_90_p_rd.set_rotation(0, 90.0, 0)
                settings.add_sensor(camera_90_p_rd)

                # LEFT STEREO +90 DEGREE SEGMENTATION
                camera_90_p_ls = Camera('90_P_LeftCameraSeg', PostProcessing='SemanticSegmentation')
                camera_90_p_ls.set_image_size(800, 600)
                camera_90_p_ls.set_position(0.27, 1.0, 1.50)
                camera_90_p_ls.set_rotation(0, 90.0, 0)
                settings.add_sensor(camera_90_p_ls)

                # RIGHT STEREO +90 DEGREE SEGMENTATION
                camera_90_p_rs = Camera('90_P_RightCameraSeg', PostProcessing='SemanticSegmentation')
                camera_90_p_rs.set_image_size(800, 600)
                camera_90_p_rs.set_position(-0.27, 1.0, 1.50)
                camera_90_p_rs.set_rotation(0, 90.0, 0)
                settings.add_sensor(camera_90_p_rs)


                '''
                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)
                '''

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)


            camera_l_to_car_transform = camera_l.get_unreal_transform()
            camera_r_to_car_transform = camera_r.get_unreal_transform()

            camera_45_n_l_to_car_transform = camera_45_n_l.get_unreal_transform()
            camera_45_n_r_to_car_transform = camera_45_n_r.get_unreal_transform()

            camera_45_p_l_to_car_transform = camera_45_p_l.get_unreal_transform()
            camera_45_p_r_to_car_transform = camera_45_p_r.get_unreal_transform()

            camera_90_n_l_to_car_transform = camera_90_n_l.get_unreal_transform()
            camera_90_n_r_to_car_transform = camera_90_n_r.get_unreal_transform()

            camera_90_p_l_to_car_transform = camera_90_p_l.get_unreal_transform()
            camera_90_p_r_to_car_transform = camera_90_p_r.get_unreal_transform()


            # Create a folder for saving episode data
            if not os.path.isdir("/data/khoshhal/Dataset/episode_{:0>4d}".format(episode)):
                os.makedirs("/data/khoshhal/Dataset/episode_{:0>4d}".format(episode))
            
            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # player_measurements = measurements.player_measurements
                world_transform = Transform(measurements.player_measurements.transform)

                # Compute the final transformation matrix.
                camera_l_to_world_transform = world_transform * camera_l_to_car_transform
                camera_r_to_world_transform = world_transform * camera_r_to_car_transform

                camera_45_n_l_to_world_transform = world_transform * camera_45_n_l_to_car_transform
                camera_45_n_r_to_world_transform = world_transform * camera_45_n_r_to_car_transform

                camera_45_p_l_to_world_transform = world_transform * camera_45_p_l_to_car_transform
                camera_45_p_r_to_world_transform = world_transform * camera_45_p_r_to_car_transform

                camera_90_n_l_to_world_transform = world_transform * camera_90_n_l_to_car_transform
                camera_90_n_r_to_world_transform = world_transform * camera_90_n_r_to_car_transform

                camera_90_p_l_to_world_transform = world_transform * camera_90_p_l_to_car_transform
                camera_90_p_r_to_world_transform = world_transform * camera_90_p_r_to_car_transform


                # Save the images to disk if requested.
                if frame >= 30 and (frame % 2 == 0):
                    if args.save_images_to_disk:
                        for name, measurement in sensor_data.items():
                            filename = args.out_filename_format.format(episode, name, (frame-30)/2)
                            measurement.save_to_disk(filename)

                        # Save Transform matrix of each camera to separated files
                        line = ""
                        
                        filename = "{}episode_{:0>4d}/LeftCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_l_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""
                        
                        filename = "{}episode_{:0>4d}/RightCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_r_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/45_N_LeftCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_45_n_l_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/45_N_RightCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_45_n_r_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/45_P_LeftCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_45_p_l_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/45_P_RightCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_45_p_r_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/90_N_LeftCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_90_n_l_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/90_N_RightCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_90_n_r_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/90_P_LeftCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_90_p_l_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""

                        filename = "{}episode_{:0>4d}/90_P_RightCamera".format(args.root_path, episode) + ".txt"
                        with open(filename, 'a') as myfile:
                            for x in np.asarray(camera_90_p_r_to_world_transform.matrix[:3, :]).reshape(-1):
                                line += "{:.8e} ".format(x)
                            line = line[:-1]
                            line += "\n"
                            myfile.write(line)
                            line = ""


                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                if not args.autopilot:

                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    control = measurements.player_measurements.autopilot_control
                    #control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

                #time.sleep(1)

            #myfile.close()
            #LeftCamera.close()
            #RightCamera.close()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '/data/khoshhal/Dataset/episode_{:0>4d}/{:s}/{:0>6d}'
    args.root_path = '/data/khoshhal/Dataset/'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
