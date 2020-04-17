for file in *.stl; 
  do "/Applications/meshlab.app/Contents/MacOS/meshlabserver" -i "$file" -o "${file%.*}.off" -s better.mlx # -om vc
done
