import cv2
import pandas as pd
from os import walk
from os import path
import numpy as np
import math


def collect_barcodes_bbox_for(directory, file, window_name, next_image_callback, save_output_callback):
    def onclick(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('point: (', x, ',', y, ')')

            # displaying the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), 1, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(x) + ',' +
                        str(y), (x + 2, y - 2), font,
                        0.5, (0, 0, 255), 1)
            cv2.imshow(window_name, img)

            # save coordinates
            saved_coords.append((x, y))

    saved_coords = []
    img = cv2.imread(directory + file, 1)
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, onclick)

    while 1:
        key = cv2.waitKey(0)
        if key == ord('c'):
            # clear saved coordinates
            saved_coords.clear()
            img = cv2.imread(directory + file, 1)
            cv2.imshow(window_name, img)
            continue
        elif key == ord('n'):
            bboxes = ''
            angles = ''
            for b in range(int(len(saved_coords) / 4)):
                coords = np.array([saved_coords[b * 4], saved_coords[b * 4 + 1], saved_coords[b * 4 + 2],
                                   saved_coords[b * 4 + 3]])
                # calulate bbox
                x_sorted_coords = coords[coords[:, 0].argsort()]
                y_sorted_coords = coords[coords[:, 1].argsort()]
                bbox = [y_sorted_coords[0][1], x_sorted_coords[0][0], y_sorted_coords[3][1], x_sorted_coords[3][0]]

                # calculate orientation angle
                point_x1 = np.array([(x_sorted_coords[0][0] + x_sorted_coords[1][0]) / 2,
                                     (x_sorted_coords[0][1] + x_sorted_coords[1][1]) / 2])
                point_x2 = np.array([(x_sorted_coords[2][0] + x_sorted_coords[3][0]) / 2,
                                     (x_sorted_coords[2][1] + x_sorted_coords[3][1]) / 2])
                point_y1 = np.array([(y_sorted_coords[0][0] + y_sorted_coords[1][0]) / 2,
                                     (y_sorted_coords[0][1] + y_sorted_coords[1][1]) / 2])
                point_y2 = np.array([(y_sorted_coords[2][0] + y_sorted_coords[3][0]) / 2,
                                     (y_sorted_coords[2][1] + y_sorted_coords[3][1]) / 2])

                if np.linalg.norm(point_x1 - point_x2) > np.linalg.norm(point_y1 - point_y2):
                    opp = point_x2[1] - point_x1[1]
                    base = point_x2[0] - point_x1[0]
                else:
                    opp = point_y2[1] - point_y1[1]
                    base = point_y2[0] - point_y1[0]

                if base == 0:
                    base = 0.0001
                angle = math.degrees(math.atan(opp / base))

                if bboxes == '':
                    bboxes = str(bbox)
                else:
                    bboxes = ',' + bboxes.join(str(bbox))

                if angles == '':
                    angles = str(angle)
                else:
                    angles = ',' + angles.join(str(angle))

            saved_coords.clear()
            data = pd.DataFrame([[file, bboxes, angles]], columns=['file', 'bounding_boxes', 'orientation_angles'])

            # send result and ask to load next image
            next_image_callback(data)
            continue
        elif key == ord('s'):
            print('save')

            save_output_callback()
            continue
        elif key == ord('q'):
            print('quit')

            # close the window
            cv2.destroyAllWindows()
            exit(0)


window = 'test-image'
images_folder = 'Muenster_Barcode_Database/N95-2592x1944_scaledTo640x480bilinear/'
output: pd.DataFrame = None
output_file = 'Muenster_Barcode_Database/annotations.csv'
_, _, image_files = next(walk(images_folder))
file_index = 0


def next_image(data: pd.DataFrame):
    print('next image')
    global output
    global file_index

    if output is None:
        output = data
    else:
        output = output.append(data, ignore_index=True)

    if file_index < len(image_files):
        file_index += 1
        collect_barcodes_bbox_for(images_folder, image_files[file_index], window, next_image, save_output)


def save_output():
    print('write output to file')
    global output

    output.to_csv(output_file, index_label='index')


# if outfile already present, then reload the output
# dataframe and continue from last saved index
if path.exists(output_file):
    output = pd.read_csv(output_file, index_col=0)
    file_index = output.tail(1).index[0]

if file_index < len(image_files):
    file_index += 1
    collect_barcodes_bbox_for(images_folder, image_files[file_index], window, next_image, save_output)
