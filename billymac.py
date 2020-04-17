import sys
import bpy
import os

#Defining directories
#when using this specify the folder path where your stl files are located
path_to_stls ="/Users/katie/Desktop/test/"
file_list = sorted(os.listdir(path_to_stls))
stl_list=[item for item in file_list if item.endswith('.stl')]
bpy.ops.object.delete(use_global=False)
for item in stl_list:
	path_to_files= os.path.join(path_to_stls, item)
#here would normally be the path imported
	bpy.ops.import_mesh.stl(filepath=path_to_files)
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.print3d_clean_non_manifold(sides=3)
	bpy.ops.mesh.print3d_check_intersect()
	bpy.ops.mesh.print3d_select_report(index=0)
	bpy.ops.mesh.delete(type='FACE')
	bpy.ops.mesh.select_all(action='TOGGLE')
	bpy.ops.mesh.print3d_clean_non_manifold(sides=3)
	

#here make a separate folder inside of the folders with the stl to export the fixed stl files #to this folder
	bpy.ops.export_mesh.stl(filepath="/Users/katie/Desktop/test/blender/",check_existing=True,axis_forward='Y',axis_up='Z',filter_glob="*.stl",use_selection=True,global_scale=1.0,use_scene_unit=False,ascii=False,use_mesh_modifiers=True,batch_mode='OBJECT')
#to run specify the file path with the blender.exe and then run -b -P "name of test"
