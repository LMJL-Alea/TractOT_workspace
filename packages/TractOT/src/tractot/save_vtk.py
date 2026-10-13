#Module produced by Nathan Decaux 

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from nibabel.affines import apply_affine

def save_vtk(streamlines, output_vtk_path,scalar_dict=None,to_lps=True): 
    """
    Save streamlines to a VTK file format with optional scalar data.

    This function converts streamlines (fiber tracts) into VTK polydata format and writes
    them to a file. Optionally, scalar values can be associated with the points of the
    streamlines.

    Parameters
    ----------
    streamlines : list of array-like
        A list of streamlines, where each streamline is an array of 3D points (x, y, z).
        Each point represents a coordinate along the streamline trajectory.
    output_vtk_path : str
        The file path where the VTK file will be saved.
    scalar_dict : dict, optional
        A dictionary mapping scalar names (str) to scalar values. The scalar values can be:
        - A numpy array of values for each point across all streamlines
        - A list of arrays that will be concatenated
        The values will be flattened and added as point data to the polydata.
        Default is None.

    Returns
    -------
    None
        The function writes the data to disk but does not return a value.
    """
    
    if to_lps:
        # ras (mm) to lps (mm)
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        streamlines=[apply_affine(to_lps, s) for s in streamlines]
        
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    point_id = 0
    for sl in streamlines:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(sl))
        for i, pt in enumerate(sl):
            points.InsertNextPoint(pt)
            line.GetPointIds().SetId(i, point_id)
            point_id += 1
        lines.InsertNextCell(line)

    polydata.SetPoints(points)
    polydata.SetLines(lines)

    if scalar_dict:
        for scalar_name, scalar_values in scalar_dict.items():
            if isinstance(scalar_values, list):
                scalar_values = np.concatenate(scalar_values)
            #vtk_array = numpy_to_vtk(scalar_values.flatten(), deep=True)
            vtk_array = numpy_to_vtk(np.vstack(scalar_values), deep=True)
            vtk_array.SetName(scalar_name)
            polydata.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    writer.SetInputData(polydata)
    writer.Write()