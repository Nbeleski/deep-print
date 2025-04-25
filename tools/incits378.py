import struct
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np

class Minutia:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation * math.pi / 180. # + math.pi / 2

class Template:
    def __init__(self, filepath):
        self.minutiae = []

        with open(filepath, 'rb') as file:
            buffer = file.read()

        if buffer is None or len(buffer) == 0:
            raise ValueError("Invalid or empty file.")

        pos = 0

        # Format Identifier
        if buffer[0:4] != b'FMR\x00':
            raise ValueError("Invalid Format Identifier.")

        # Version of the standard
        if buffer[4:8] != b' 20\x00':
            raise ValueError("Unsupported version of the standard.")

        # Length of total record in bytes
        length_of_record = 0
        if buffer[8] == 0x00 and buffer[9] == 0x00:
            length_of_record = struct.unpack('>I', buffer[10:14])[0]
            pos = 14
        else:
            length_of_record = struct.unpack('>H', buffer[8:10])[0]
            pos = 10

        if length_of_record == 32:
            raise ValueError("Null template detected.")

        # Skip Capture System
        pos += 6

        # Image Size
        image_width = struct.unpack('>H', buffer[pos:pos + 2])[0]
        pos += 2
        image_height = struct.unpack('>H', buffer[pos:pos + 2])[0]
        pos += 2

        # Image Resolution
        horizontal_resolution = struct.unpack('>H', buffer[pos:pos + 2])[0]
        horizontal_resolution = int(horizontal_resolution * 2.54 + 0.5)
        pos += 2
        vertical_resolution = struct.unpack('>H', buffer[pos:pos + 2])[0]
        vertical_resolution = int(vertical_resolution * 2.54 + 0.5)
        pos += 2

        # Skip Number of Finger Views and Reserved Byte
        pos += 2

        # Finger View Processing
        pos += 1  # Skip finger position
        imp = buffer[pos] & 0x0F
        pos += 1

        finger_quality = buffer[pos]
        pos += 1

        # NumMinutiae
        num_minutiae = buffer[pos]
        pos += 1

        # Extract Minutiae
        for _ in range(num_minutiae):
            minutia_type = (buffer[pos] >> 6) & 0x03
            pos_x = (buffer[pos] & 0x3F) << 8 | buffer[pos + 1]
            pos_y = (buffer[pos + 2] & 0x3F) << 8 | buffer[pos + 3]
            orientation = buffer[pos + 4] * 2
            quality = buffer[pos + 5]

            self.minutiae.append(Minutia(pos_x, pos_y, orientation))

            pos += 6

        # After processing minutiae, process the extended block if present.
        if pos < len(buffer):
            # Read the total extension block length (2 bytes, big-endian)
            total_extension_length = (buffer[pos] << 8) | buffer[pos + 1]
            pos += 2
        
            # # Loop through each extension in the block.
            # while pos < len(buffer):
            #     # Read the extension type (2 bytes, big-endian)
            #     extension_type = (buffer[pos] << 8) | buffer[pos + 1]
            #     pos += 2
        
            #     # Read the length of the current extension (2 bytes, big-endian)
            #     current_extension_length = (buffer[pos] << 8) | buffer[pos + 1]
            #     pos += 2
        
            #     if extension_type == 1:
            #         # Extension type 1 contains ridge information (not implemented)
            #         pos += current_extension_length - 4
        
            #     elif extension_type == 2:
            #         # Extension type 2: core and delta information.
            #         if current_extension_length < 2:
            #             raise ValueError("Erro no tamanho dos campos de extensão")
        
            #         # Process core information.
            #         # The first byte indicates (a) whether cores include an angle and (b) the number of cores.
            #         core_byte = buffer[pos]
            #         core_has_angle = bool(core_byte & 0b01000000)
            #         core_num = core_byte & 0b00001111
            #         pos += 1
        
            #         # Initialize the list to store core data.
            #         self.cores = []
            #         for _ in range(core_num):
            #             # Each core's x-coordinate is encoded in two parts:
            #             #   high 6 bits in buffer[pos] and low 8 bits in buffer[pos+1]
            #             x = ((buffer[pos] & 0x3F) << 8) | buffer[pos + 1]
            #             # Similarly for y:
            #             y = ((buffer[pos + 2] & 0x3F) << 8) | buffer[pos + 3]
            #             angle = 0
            #             if core_has_angle:
            #                 angle = buffer[pos + 4]
            #                 pos += 5
            #             else:
            #                 pos += 4
            #             self.cores.append(Core(x, y, angle))
        
            #         # Process delta information.
            #         # The next byte encodes if deltas include angles and how many deltas there are.
            #         delta_byte = buffer[pos]
            #         delta_has_angle = bool(delta_byte & 0b01000000)
            #         delta_num = delta_byte & 0b00001111
            #         pos += 1
        
            #         # Initialize the list to store delta data.
            #         self.deltas = []
            #         for _ in range(delta_num):
            #             x = ((buffer[pos] & 0x3F) << 8) | buffer[pos + 1]
            #             y = ((buffer[pos + 2] & 0x3F) << 8) | buffer[pos + 3]
            #             if delta_has_angle:
            #                 ang1 = buffer[pos + 4]
            #                 ang2 = buffer[pos + 5]
            #                 ang3 = buffer[pos + 6]
            #                 pos += 7
            #             else:
            #                 ang1 = ang2 = ang3 = 0
            #                 pos += 4
            #             self.deltas.append(Delta(x, y, ang1, ang2, ang3))
        
            #     elif extension_type == 3:
            #         # Extension type 3 contains proprietary group information.
            #         pos += current_extension_length - 4
        
            #     else:
            #         # For any other (unknown) extension types, skip the extension data.
            #         pos += current_extension_length - 4

    def draw_minutiae(self, image_width=512, image_height=512, radius=3, color=(255, 0, 0), thickness=2):
        # Create a blank image
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        # Draw all minutiae points on the image
        for minutia in self.minutiae:
            cv2.circle(image, (minutia.x, minutia.y), radius, color, thickness)
            ori = minutia.orientation + math.pi / 2.
            cv2.line(image, (minutia.x, minutia.y), (int(minutia.x + 12 * math.sin(ori)), int(minutia.y + 12 * math.cos(ori))), (255, 0, 0), 2)

        return image