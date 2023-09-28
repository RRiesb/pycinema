
import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
ParameterView_0 = vf1.insertView( 0, pycinema.theater.views.ParameterView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
ColorMappingView_0 = vf1.insertView( 2, pycinema.theater.views.ColorMappingView() )
vf1.setSizes([128, 590, 383])
vf2 = vf0.insertFrame(1)
vf2.setVerticalOrientation()
vf3 = vf2.insertFrame(0)
vf3.setHorizontalOrientation()
ImageView_0 = vf3.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_2 = vf3.insertView( 1, pycinema.theater.views.ImageView() )
vf3.setSizes([839, 839])
vf2.insertView( 1, pycinema.theater.views.NodeView() )
vf2.setSizes([553, 553])
vf0.setSizes([232, 1683])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()
ShaderLineAO_0 = pycinema.filters.ShaderLineAO()
ShaderPointAO_0 = pycinema.filters.ShaderPointAO()
ImageAnnotation_1 = pycinema.filters.ImageAnnotation()

# properties
ParameterView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ParameterView_0.inputs.ignore.set(['file', 'id'], False)
ParameterView_0.inputs.state.set({'Phi': {'C': False, 'T': 'S', 'S': 0, 'O': [0]}, 'Theta': {'C': False, 'T': 'S', 'S': 0, 'O': [0]}}, False)
TableView_0.inputs.table.set(ParameterView_0.outputs.table, False)
ColorMappingView_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ColorMappingView_0.inputs.channel.set("Elevation", False)
ColorMappingView_0.inputs.map.set("plasma", False)
ColorMappingView_0.inputs.range.set((0, 1), False)
ColorMappingView_0.inputs.nan.set((1, 1, 1, 1), False)
ColorMappingView_0.inputs.composition_id.set(-1, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
CinemaDatabaseReader_0.inputs.path.set("E:/BA/pyC/pycinema/data/Oak.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(ParameterView_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.images_b.set([], False)
DepthCompositing_0.inputs.depth_channel.set("depth", False)
DepthCompositing_0.inputs.compose.set(ParameterView_0.outputs.compose, False)
ImageAnnotation_0.inputs.images.set(ShaderPointAO_0.outputs.images, False)
ImageAnnotation_0.inputs.xy.set((20, 20), False)
ImageAnnotation_0.inputs.size.set(20, False)
ImageAnnotation_0.inputs.spacing.set(0, False)
ImageAnnotation_0.inputs.color.set((), False)
ImageAnnotation_0.inputs.ignore.set(['file', 'id'], False)
ShaderLineAO_0.inputs.images.set(ColorMappingView_0.outputs.images, False)
ShaderLineAO_0.inputs.radius.set(1.5, False)
ShaderLineAO_0.inputs.samples.set(60, False)
ShaderLineAO_0.inputs.scalers.set(2, False)
ShaderLineAO_0.inputs.densityweight.set(1.0, False)
ShaderLineAO_0.inputs.totalStrength.set(1.0, False)
ShaderPointAO_0.inputs.images.set(ColorMappingView_0.outputs.images, False)
ShaderPointAO_0.inputs.radius.set(1.5, False)
ShaderPointAO_0.inputs.samples.set(60, False)
ShaderPointAO_0.inputs.scalers.set(2, False)
ShaderPointAO_0.inputs.densityweight.set(1.0, False)
ShaderPointAO_0.inputs.totalStrength.set(1.0, False)
ImageAnnotation_1.inputs.images.set(ShaderLineAO_0.outputs.images, False)
ImageAnnotation_1.inputs.xy.set((20, 20), False)
ImageAnnotation_1.inputs.size.set(20, False)
ImageAnnotation_1.inputs.spacing.set(0, False)
ImageAnnotation_1.inputs.color.set((), False)
ImageAnnotation_1.inputs.ignore.set(['file', 'id'], False)
ImageView_2.inputs.images.set(ImageAnnotation_1.outputs.images, False)

# execute pipeline
ParameterView_0.update()