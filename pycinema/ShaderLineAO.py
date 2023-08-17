from .Shader import *
import cv2
import numpy

# TODO: actually implement the shader, change the parameters to the ones we need
class ShaderLineAO(Shader):
    def __init__(self):
        super().__init__(['rgbaTex','depthTex', 'noiseTex'])
        self.addInputPort("images", [])
        self.addInputPort("radius", 1.5)
        self.addInputPort("samples", 32)
        self.addInputPort("scalers", 3)
        self.addOutputPort("images", [])

    def getFragmentShaderCode(self):
        return """
#version 330

uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform sampler2D noiseTex;
uniform float radius;

uniform float diff_area;
uniform int samples;
uniform vec2 resolution;
uniform int scalers;

//uniform int currentLevel;

//for zoom function
uniform float lineWidth = 2.0;

float radiusSS = radius;


//influence of SSAO
uniform float totalStrength = 1.0;


in vec2 uv;
out vec4 color;



//depth

float getDepth(vec2 where)
{
    return texture(depthTex, where).r;
}

float getDepth(vec2 where, float lod)
{
    //TODO Gauss  dl(P) depth map
    return textureLod(depthTex, where, lod).r;
}

//for randNormal
vec2 getTextureSize(sampler2D texture)
{
    vec2 textureSize = textureSize(texture, 0);
    return textureSize;
}


//Normal
//TODO Normale für Gauss

vec3 computeNormal(in vec2 uv, in float depth){
  vec2 pixelSize = 1./resolution;
  vec3 eps = vec3( pixelSize.x, pixelSize.y, 0 );
  float depthN = getDepth(uv.xy + eps.zy);
  float depthS = getDepth(uv.xy - eps.zy);
  float depthE = getDepth(uv.xy + eps.xz);
  float depthW = getDepth(uv.xy - eps.xz);
  // vec3 dx = vec3(2.0*eps.xz,depthE-depthW);
  // vec3 dy = vec3(2.0*eps.zy,depthN-depthS);
  vec3 dx = vec3(eps.xz, abs(depth-depthW) < abs(depth-depthE)
    ? depthW-depth
    : depth-depthE
  );
  vec3 dy = vec3(eps.zy, abs(depth-depthN) < abs(depth-depthS)
    ? depth-depthN
    : depthS-depth
  );
  return normalize(cross(dx, dy));
}

float getZoom()
{
    return texture( rgbaTex, uv ).r;
}

vec3 getTangent(vec2 hemispherePoint, float lod) {
    // Berechnen von z basierend auf x und y
    float z = sqrt(1.0 - hemispherePoint.x * hemispherePoint.x - hemispherePoint.y * hemispherePoint.y);

    // LOD eingefügen
    // ...

    // Rückgabe des Tangentenvektors
    return vec3(hemispherePoint.x, hemispherePoint.y, z);
}

///////////////////////////////////////////
//  LineAO
//////////////////////////////////////////
float computeLineAO(vec2 coord){


    vec4 rgba = texture(rgbaTex,coord);

    vec3 lightPos = vec3(0,10,10);


    //openwalnut

    // Fall-off for SSAO per occluder
    const float falloff = 0.00001;



    //random normal for reflecting sample rays
    vec2 noiseSize = getTextureSize(noiseTex);
    //vec2 randCoordsNorm = coord * noiseSize.x;
    //vec3 randNormal = vec3(fract(sin(dot(randCoordsNorm, vec2(12.9898, 78.233))) * 43758.5453),
    //                    fract(sin(dot(randCoordsNorm, vec2(23.123, 45.678))) * 98365.1234),
    //                    fract(sin(dot(randCoordsNorm, vec2(34.567, 89.012))) * 29384.9876));
    //randNormal = normalize(randNormal * 2.0 - vec3(1.0));

    vec3 randNormal = normalize( ( texture( noiseTex, coord * int(noiseSize.x) ).xyz * 2.0 ) - vec3( 1.0 ) );


    //current pixels normal and depth
    //vec3 currentPixelSample = getNormal( coord ).xyz;
    float currentPixelDepth = getDepth( coord );
    vec3 currentPixelSample = computeNormal( coord, getDepth(coord) ).xyz;

    //current fragment coords
    vec3 ep = vec3( coord.xy, currentPixelDepth);

    //normal of current fragment
    vec3 normal = currentPixelSample.xyz;

    //invariant for zooming

    float maxPixels = max( float( resolution.x ), float( resolution.y ) );
    //radiusSS = ( getZoom() * radius / maxPixels ) / (1.0 - currentPixelDepth );
    radiusSS = ( radius / maxPixels ) / (1.0 - currentPixelDepth );

    //some temoraries needed inside the loop
    vec3 ray;
    vec3 hemispherePoint;
    vec3 occluderNormal;

    float occluderDepth;
    float depthDifference;
    float normalDifference;

    float occlusion = 0.0;
    float radiusScaler = 0.0;

    //sample for different radii
    for( int l = 0; l < scalers; ++l)
    {
        float occlusionStep = 0.0;

        //diffrent from paper
        radiusScaler += 1.5 + l;

        //get samples and check for Occluders
        int numSamplesAdded = 0;

        for( int i = 0; i < samples; ++i){

            //random normal from noise texture

            //vec2 randCoordsSphere = vec2(float(i) / float(samples), float(l + 1) / float(scalers));
            //vec3 randSphereNormal = vec3(fract(sin(dot(randCoordsSphere,  vec2(12.9898, 78.233))) * 43758.5453),
            //                            fract(sin(dot(randCoordsSphere, vec2(23.123, 45.678))) * 98365.1234),
            //                            fract(sin(dot(randCoordsSphere, vec2(34.567, 89.012))) * 29384.9876));
            //randSphereNormal = normalize(randSphereNormal * 2.0 - vec3(1.0));


            vec3 randSphereNormal = ( texture( noiseTex, vec2( float( i ) / float( samples ), float( l + 1 ) / float( scalers ) ) ).rgb * 2.0 ) - vec3( 1.0 );


            vec3 hemisphereVector = reflect( randSphereNormal, randNormal );
            ray = radiusScaler * radiusSS * hemisphereVector;
            ray = sign( dot( ray, normal ) ) * ray;

            //point in texture space on the hemisphere
            hemispherePoint = ray + ep;

            if( ( hemispherePoint.x < 0.0 ) || ( hemispherePoint.x > 1.0 ) ||
                ( hemispherePoint.y < 0.0 ) || ( hemispherePoint.y > 1.0 ) )
            {
                continue;
            }

            //count Samples used
            numSamplesAdded++;

            float lod = 0.0;

            //gausspyramid for Level of Detail
            //lod = float(l);

            //occluderDepth = getDepth( hemispherePoint.xy, lod );
            occluderDepth = getDepth( hemispherePoint.xy );
            //occluderNormal = getNormal( hemispherePoint.xy).xyz;
            occluderNormal = computeNormal( hemispherePoint.xy, occluderDepth).xyz;
            depthDifference = currentPixelDepth - occluderDepth;

            //difference between the normals as a weight -> how much occludes fragment
            float pointDiffuse = max( dot( hemisphereVector, normal ), 0.0 );

            //spielt Rolle bei brightness:
            //diffuse reflected light
            //#ifdef OccluderLight
            //vec3 t= getTangent( hemispherePoint.xy, lod ).xyz;
            //vec3 newnorm = normalize( cross( normalize( cross( t, normalize( hemisphereVector ) ) ), t ) );
            //float occluderDiffuse = max( dot( newnorm, lightPos.xyz ), 0.0);
            //#else

            //disable effect
            float occluderDiffuse = 0.0;
            //#endif


            //specular reflection
            vec3 H = normalize( lightPos.xyz + normalize( hemisphereVector ) );
            float occluderSpecular = pow( max( dot( H, occluderNormal ), 0.0 ), 100.0);

            normalDifference = pointDiffuse * ( occluderSpecular + occluderDiffuse );
            normalDifference = 1.5 - normalDifference;


            //shadowiness

            float SCALER = 1.0 - ( l / (float (scalers - 1.0 ) ) );
            float densityInfluence = SCALER * SCALER;

            float densityWeight = 1.0 - smoothstep( falloff, densityInfluence, depthDifference );


            // reduce occlusion if occluder is far away
            occlusionStep += normalDifference * densityWeight * step( falloff, depthDifference );
            //occlusionStep = 0.0;
        }

        occlusion += ( 1.0 / float( numSamplesAdded ) ) * occlusionStep;
    }

    float occlusionScalerFactor = 1.0 / ( scalers );
    occlusionScalerFactor *= totalStrength;

    //output result
    return clamp( ( 1.0 - ( occlusionScalerFactor * occlusion  ) ), 0.0 , 1.0 );
}

void main(){

    vec4 rgba = texture(rgbaTex, uv);
    float c = computeLineAO(uv);

    //color = vec4(c, rgba.a);
    color = vec4(mix( vec3(0), rgba.rgb, c), rgba.a);
    //color = vec4(rgba.rgb*c, rgba.a);
    //color = vec4(vec3(rgba.rgb*c), 1.0);
}

"""

    def render(self,image):
        # update texture
        self.rgbaTex.write(image.channels['rgba'].tobytes())
        self.depthTex.write(image.channels['depth'].tobytes())

        # render
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # read framebuffer
        outImage = image.copy()
        outImage.channels['rgba'] = self.readFramebuffer()

        return outImage

    # def downsample(self,image):
        image_data= numpy.array(image)
        # Verkleinert das Bild auf die Hälfte seiner Größe
        return cv2.pyrDown(image_data)

    # def create_gaussian_pyramid(self, image, levels):
        highres = image.copy()
        pyramid = [highres]
        for i in range(levels):
            # aktuelle Level Gauß-Pyramide
            self.program['currentLevel'].value = i
            # Skaliere das Bild herunter
            lowres = cv2.pyrDown(highres)
            highres = cv2.pyrUp(lowres)
            pyramid.append(highres)
        return pyramid

    def _update(self):
        results = []

        images = self.inputs.images.get()
        if len(images)<1:
            self.outputs.images.set(results)
            return 1

        # first image
        image0 = images[0]
        if not 'depth' in image0.channels or not 'rgba' in image0.channels:
            self.outputs.images.set(images)
            return 1

        shape = image0.channels['rgba'].shape
        if len(shape)!=3:
            shape = (shape[0],shape[1],1)
        res = shape[:2][::-1]

        # init framebuffer
        self.initFramebuffer(res)

        # set uniforms

        self.program['radius'].value = float(self.inputs.radius.get())
        self.program['samples'].value = int(self.inputs.samples.get())
        self.program['scalers'].value = int(self.inputs.scalers.get())
        self.program['resolution'].value = res





        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')
        self.depthTex = self.createTexture(1,res,1,dtype='f4')
        self.noiseTex = self.createNoiseTexture(2, (64,64), 3)


        # TODO richtig einbinden
        # images = self.create_gaussian_pyramid(images,4)


        for image in images:

            results.append( self.render(image) )



        self.rgbaTex.release()
        self.depthTex.release()
        self.noiseTex.release()

        self.outputs.images.set(results)

        return 1
