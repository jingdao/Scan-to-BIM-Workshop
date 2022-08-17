import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import uuid
import time
import tempfile
import ifcopenshell

def draw_geometries(geometries, names=None, show_axes=False):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors), name=None if names is None else names[len(graph_objects)])
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)
            
            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)
        
    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=show_axes),
                yaxis=dict(visible=show_axes),
                zaxis=dict(visible=show_axes),
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=1),
            )
        )
    )
    fig.show()

# Creates an IfcAxis2Placement3D from Location, Axis and RefDirection specified as Python tuples
def create_ifcaxis2placement(ifcfile, point=(0.,0.,0.), dir1=(0.,0.,1.), dir2=(1.,0.,0.)):
    point = ifcfile.createIfcCartesianPoint(point)
    dir1 = ifcfile.createIfcDirection(dir1)
    dir2 = ifcfile.createIfcDirection(dir2)
    axis2placement = ifcfile.createIfcAxis2Placement3D(point, dir1, dir2)
    return axis2placement

# Creates an IfcLocalPlacement from Location, Axis and RefDirection, specified as Python tuples, and relative placement
def create_ifclocalplacement(ifcfile, point=(0.,0.,0.), dir1=(0.,0.,1.), dir2=(1.,0.,0.), relative_to=None):
    axis2placement = create_ifcaxis2placement(ifcfile,point,dir1,dir2)
    ifclocalplacement2 = ifcfile.createIfcLocalPlacement(relative_to,axis2placement)
    return ifclocalplacement2

# Creates an IfcPolyLine from a list of points, specified as Python tuples
def create_ifcpolyline(ifcfile, point_list):
    ifcpts = []
    for point in point_list:
        point = ifcfile.createIfcCartesianPoint(point)
        ifcpts.append(point)
    polyline = ifcfile.createIfcPolyLine(ifcpts)
    return polyline
    
# Creates an IfcExtrudedAreaSolid from a list of points, specified as Python tuples
def create_ifcextrudedareasolid(ifcfile, point_list, ifcaxis2placement, extrude_dir, extrusion):
    polyline = create_ifcpolyline(ifcfile, point_list)
    ifcclosedprofile = ifcfile.createIfcArbitraryClosedProfileDef("AREA", None, polyline)
    ifcdir = ifcfile.createIfcDirection(extrude_dir)
    ifcextrudedareasolid = ifcfile.createIfcExtrudedAreaSolid(ifcclosedprofile, ifcaxis2placement, ifcdir, extrusion)
    return ifcextrudedareasolid
    

def writeIFC(wall_lines, filename):
    create_guid = lambda: ifcopenshell.guid.compress(uuid.uuid1().hex)

    # IFC template creation
    timestamp = time.time()
    timestring = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp))
    creator = ""
    organization = ""
    application, application_version = "IfcOpenShell", "0.6"
    project_globalid, project_name = create_guid(), filename
        
    # A template IFC file to quickly populate entity instances for an IfcProject with its dependencies
    template = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('%(filename)s','%(timestring)s',('%(creator)s'),('%(organization)s'),'%(application)s','%(application)s','');
FILE_SCHEMA(('IFC2X3'));
ENDSEC;
DATA;
#1=IFCPERSON($,$,'%(creator)s',$,$,$,$,$);
#2=IFCORGANIZATION($,'%(organization)s',$,$,$);
#3=IFCPERSONANDORGANIZATION(#1,#2,$);
#4=IFCAPPLICATION(#2,'%(application_version)s','%(application)s','');
#5=IFCOWNERHISTORY(#3,#4,$,.ADDED.,$,#3,#4,%(timestamp)s);
#6=IFCDIRECTION((1.,0.,0.));
#7=IFCDIRECTION((0.,0.,1.));
#8=IFCCARTESIANPOINT((0.,0.,0.));
#9=IFCAXIS2PLACEMENT3D(#8,#7,#6);
#10=IFCDIRECTION((0.,1.,0.));
#11=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#9,#10);
#12=IFCDIMENSIONALEXPONENTS(0,0,0,0,0,0,0);
#13=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#14=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);
#15=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);
#16=IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.);
#17=IFCMEASUREWITHUNIT(IFCPLANEANGLEMEASURE(0.017453292519943295),#16);
#18=IFCCONVERSIONBASEDUNIT(#12,.PLANEANGLEUNIT.,'DEGREE',#17);
#19=IFCUNITASSIGNMENT((#13,#14,#15,#18));
#20=IFCPROJECT('%(project_globalid)s',#5,'%(project_name)s',$,$,$,$,(#11),#19);
ENDSEC;
END-ISO-10303-21;
    """ % locals()

    # Write the template to a temporary file 
    temp_handle, temp_filename = tempfile.mkstemp(suffix=".ifc")
    with open(temp_filename, "wb") as f:
        f.write(template.encode())
     
    # Obtain references to instances defined in template
    ifcfile = ifcopenshell.open(temp_filename)
    owner_history = ifcfile.by_type("IfcOwnerHistory")[0]
    project = ifcfile.by_type("IfcProject")[0]
    context = ifcfile.by_type("IfcGeometricRepresentationContext")[0]

    # IFC hierarchy creation
    site_placement = create_ifclocalplacement(ifcfile)
    site = ifcfile.createIfcSite(create_guid(), owner_history, "Site", None, None, site_placement, None, None, "ELEMENT", None, None, None, None, None)

    building_placement = create_ifclocalplacement(ifcfile, relative_to=site_placement)
    building = ifcfile.createIfcBuilding(create_guid(), owner_history, 'Building', None, None, building_placement, None, None, "ELEMENT", None, None, None)

    storey_placement = create_ifclocalplacement(ifcfile, relative_to=building_placement)
    elevation = 0.0
    building_storey = ifcfile.createIfcBuildingStorey(create_guid(), owner_history, "Storey", None, None, storey_placement, None, None, "ELEMENT", elevation)

    container_storey = ifcfile.createIfcRelAggregates(create_guid(), owner_history, "Building Container", None, building, [building_storey])
    container_site = ifcfile.createIfcRelAggregates(create_guid(), owner_history, "Site Container", None, site, [building])
    container_project = ifcfile.createIfcRelAggregates(create_guid(), owner_history, "Project Container", None, project, [site])

    # Wall creation: Define the wall shape as a polyline axis and an extruded area solid
    elements = []
    for i in range(len(wall_lines)): # [x0, y0, length, direction, height, thickness]
        line = wall_lines[i]
        h = float(line[4]) #element height(M)
        t = float(line[5]) #element thickness(M)
#        l = float(line[2]) + t #element lenght(M)
        l = float(line[2])

#        x0 = float(line[0]) - t/2
        x0 = float(line[0])
        y0 = float(line[1])

        dir1 = (0.0, 0.0, 1.0) #2D direction1
        dir2 = (1.0, 0.0, 0.0) #2D direction2(x)
        if line[3]=='y':
            dir2 = (0.0, 1.0, 0.0)

            x0 = float(line[0])
#            y0 = float(line[1]) - t/2
            y0 = float(line[1])

        solid_dir = (0.0, 0.0, 1.0)
        s0 = (x0, y0, 0.0) #start position(x,y,z)
        
        wall_placement = create_ifclocalplacement(ifcfile, relative_to=storey_placement)
        polyline = create_ifcpolyline(ifcfile, [s0, (l, 0.0, 0.0)])
        axis_representation = ifcfile.createIfcShapeRepresentation(context, "Axis", "Curve2D", [polyline])

        extrusion_placement = create_ifcaxis2placement(ifcfile, s0, dir1, dir2)
        point_list_extrusion_area = [(0.0, -t/2, 0.0), (l, -t/2, 0.0), (l, t/2, 0.0), (0.0, t/2, 0.0), (0.0, -t/2, 0.0)]
        solid = create_ifcextrudedareasolid(ifcfile, point_list_extrusion_area, extrusion_placement, solid_dir, h)
        body_representation = ifcfile.createIfcShapeRepresentation(context, "Body", "SweptSolid", [solid])

        product_shape = ifcfile.createIfcProductDefinitionShape(None, None, [axis_representation, body_representation])
        wall = ifcfile.createIfcWallStandardCase(create_guid(), owner_history, "Wall", "wall", None, wall_placement, product_shape, None)
        elements.append(wall)

        # Define and associate the wall material
        material = ifcfile.createIfcMaterial("wall material")
        material_layer = ifcfile.createIfcMaterialLayer(material, t, None) #t=wall tickness
        material_layer_set = ifcfile.createIfcMaterialLayerSet([material_layer], None)
        material_layer_set_usage = ifcfile.createIfcMaterialLayerSetUsage(material_layer_set, "AXIS2", "POSITIVE", -0.1)
        ifcfile.createIfcRelAssociatesMaterial(create_guid(), owner_history, RelatedObjects=[wall], RelatingMaterial=material_layer_set_usage)

        # Create and assign property set
        property_values = [
            ifcfile.createIfcPropertySingleValue("Reference", "Reference", ifcfile.create_entity("IfcText", "Describe the Reference"), None),
            ifcfile.createIfcPropertySingleValue("IsExternal", "IsExternal", ifcfile.create_entity("IfcBoolean", True), None),
            ifcfile.createIfcPropertySingleValue("ThermalTransmittance", "ThermalTransmittance", ifcfile.create_entity("IfcReal", 2.569), None),
            ifcfile.createIfcPropertySingleValue("IntValue", "IntValue", ifcfile.create_entity("IfcInteger", 2), None)
        ]
        property_set = ifcfile.createIfcPropertySet(create_guid(), owner_history, "Pset_WallCommon", None, property_values)
        ifcfile.createIfcRelDefinesByProperties(create_guid(), owner_history, None, None, [wall], property_set)

        # Add quantity information
        quantity_values = [
            ifcfile.createIfcQuantityLength("Length", "Length of the wall", None, l),
            ifcfile.createIfcQuantityArea("Area", "Area of the front face", None, l * solid.Depth),
            ifcfile.createIfcQuantityVolume("Volume", "Volume of the wall", None, l * solid.Depth * material_layer.LayerThickness)
        ]
        wall_quantity = ifcfile.createIfcElementQuantity(create_guid(), owner_history, "BaseQuantities", None, None, quantity_values)
        ifcfile.createIfcRelDefinesByProperties(create_guid(), owner_history, None, None, [wall], wall_quantity)
        # Relate the wall to the building storey
        ifcfile.createIfcRelContainedInSpatialStructure(create_guid(), owner_history, "Building Storey Container", None, [wall], building_storey)

    # Write the contents of the file to disk
    ifcfile.write(filename)