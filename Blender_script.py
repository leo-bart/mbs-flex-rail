import bpy
import csv

# Set the file paths
stl_file_path = "/home/leonardo/git/mbs-flex-rail/mbs-flex-rail//Rodeiro montado.stl"
csv_file_path = "/home/leonardo/teste.csv"

# Import STL geometry
bpy.ops.import_mesh.stl(filepath=stl_file_path)

# Select the imported object
imported_obj = bpy.context.selected_objects[0]

# Create an animation timeline based on the CSV data
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        # Assuming your CSV has 'time', 'x', 'y', and 'z' columns
        time = float(row['Time'])*1000
        x_translation = float(row['x'])
        y_translation = float(row['y'])
        z_translation = float(row['z'])

        # Set the frame based on time
        frame = int(time * bpy.context.scene.render.fps)

        # Set the location of the object at the specified frame
        bpy.context.scene.frame_set(frame)
        imported_obj.location.x = x_translation
        imported_obj.location.y = y_translation
        imported_obj.location.z = z_translation

        # Keyframe the location
        imported_obj.keyframe_insert(data_path='location', frame=frame)

# Set the end frame of the animation (adjust this based on your needs)
end_frame = bpy.context.scene.frame_end
bpy.context.scene.frame_end = end_frame

# Set the timeline to start at frame 1
bpy.context.scene.frame_start = 1

# Set the playback range to match the animation range
bpy.context.scene.frame_preview_start = 1
bpy.context.scene.frame_preview_end = end_frame

# Set the current frame to the start frame
bpy.context.scene.frame_set(1)