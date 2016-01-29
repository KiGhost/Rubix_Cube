/* SCE CONFIDENTIAL
 * PlayStation(R)Vita Programmer Tool Runtime Library Release 02.000.081
 * Copyright (C) 2010 Sony Computer Entertainment Inc.
 * All Rights Reserved.
 */

// All fonts related stuff has been ripped out.

/*	

	This sample shows how to initialize libdbgfont (and libgxm),
	and render debug font with triangle for clear the screen.

	This sample is split into the following sections:

		1. Initialize libdbgfont
		2. Initialize libgxm
		3. Allocate display buffers, set up the display queue
		4. Create a shader patcher and register programs
		5. Create the programs and data for the clear
		6. Start the main loop
			7. Update step
			8. Rendering step
			9. Flip operation and render debug font at display callback
		10. Wait for rendering to complete
		11. Destroy the programs and data for the clear triangle
		12. Finalize libgxm
		13. Finalize libdbgfont

	Please refer to the individual comment blocks for details of each section.
*/
#include <algorithm>	// std::swap
#include <vector>		// std::vector
#include <iostream>

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sceerror.h>

#include <gxm.h>
#include <kernel.h>
#include <touch.h>			// Touch
#include <ctrl.h>			// Control
#include <sce_geometry.h>	// Ray and Planes for collision detection
#include <display.h>
#include <libdbg.h>

#include <libdbgfont.h>
#include <math.h>

#include <vectormath.h>
using namespace sce::Vectormath::Simd::Aos;


/*	Define the debug font pixel color format to render to. */
#define DBGFONT_PIXEL_FORMAT		SCE_DBGFONT_PIXELFORMAT_A8B8G8R8


/*	Define the width and height to render at the native resolution */ 
#define DISPLAY_WIDTH				960
#define DISPLAY_HEIGHT				544
#define DISPLAY_STRIDE_IN_PIXELS	1024

/*	Define the libgxm color format to render to.
	This should be kept in sync with the display format to use with the SceDisplay library.
*/
#define DISPLAY_COLOR_FORMAT		SCE_GXM_COLOR_FORMAT_A8B8G8R8
#define DISPLAY_PIXEL_FORMAT		SCE_DISPLAY_PIXELFORMAT_A8B8G8R8

/*	Define the number of back buffers to use with this sample.  Most applications
	should use a value of 2 (double buffering) or 3 (triple buffering).
*/
#define DISPLAY_BUFFER_COUNT		3

/*	Define the maximum number of queued swaps that the display queue will allow.
	This limits the number of frames that the CPU can get ahead of the GPU,
	and is independent of the actual number of back buffers.  The display
	queue will block during sceGxmDisplayQueueAddEntry if this number of swaps
	have already been queued.
*/
#define DISPLAY_MAX_PENDING_SWAPS	2


/*	Helper macro to align a value */
#define ALIGN(x, a)					(((x) + ((a) - 1)) & ~((a) - 1))

// Rad
#define PI							3.14159265358979323846f
#define RAD_ROTATION				(PI / 180)

/*	The build process for the sample embeds the shader programs directly into the
	executable using the symbols below.  This is purely for convenience, it is
	equivalent to simply load the binary file into memory and cast the contents
	to type SceGxmProgram.
*/
extern const SceGxmProgram binaryClearVGxpStart;
extern const SceGxmProgram binaryClearFGxpStart;

/*	Data structure for clear geometry */
typedef struct ClearVertex
{
	float x;
	float y;
} ClearVertex;

// !! Data related to rendering vertex.
extern const SceGxmProgram binaryBasicVGxpStart;
extern const SceGxmProgram binaryBasicFGxpStart;

/*	Data structure for basic geometry */
typedef struct BasicVertex
{
	float position[3];		// 0 = x, 1 = y, 2 = z
	uint32_t color;			// Data gets expanded to float 4 in vertex shader.
	float normal[3];		// 0 = x, 1 = y, 2 = z
	float uv[2];
} BasicVertex;

/*	Data structure to pass through the display queue.  This structure is
	serialized during sceGxmDisplayQueueAddEntry, and is used to pass
	arbitrary data to the display callback function, called from an internal
	thread once the back buffer is ready to be displayed.

	In this example, we only need to pass the base address of the buffer.
*/
typedef struct DisplayData
{
	void *address;
} DisplayData;

static SceGxmContextParams		s_contextParams;			/* libgxm context parameter */
static SceGxmRenderTargetParams s_renderTargetParams;		/* libgxm render target parameter */
static SceGxmContext			*s_context			= NULL;	/* libgxm context */
static SceGxmRenderTarget		*s_renderTarget		= NULL;	/* libgxm render target */
static SceGxmShaderPatcher		*s_shaderPatcher	= NULL;	/* libgxm shader patcher */

/*	display data */
static void							*s_displayBufferData[ DISPLAY_BUFFER_COUNT ];
static SceGxmSyncObject				*s_displayBufferSync[ DISPLAY_BUFFER_COUNT ];
static int32_t						s_displayBufferUId[ DISPLAY_BUFFER_COUNT ];
static SceGxmColorSurface			s_displaySurface[ DISPLAY_BUFFER_COUNT ];
static uint32_t						s_displayFrontBufferIndex = 0;
static uint32_t						s_displayBackBufferIndex = 0;
static SceGxmDepthStencilSurface	s_depthSurface;

/*	shader data */
static int32_t					s_clearVerticesUId;
static int32_t					s_clearIndicesUId;
static SceGxmShaderPatcherId	s_clearVertexProgramId;
static SceGxmShaderPatcherId	s_clearFragmentProgramId;
// !! Shader patcher added.
static SceGxmShaderPatcherId	s_basicVertexProgramId;
static SceGxmShaderPatcherId	s_basicFragmentProgramId;
static SceUID					s_patcherFragmentUsseUId;
static SceUID					s_patcherVertexUsseUId;
static SceUID					s_patcherBufferUId;
static SceUID					s_depthBufferUId;
static SceUID					s_vdmRingBufferUId;
static SceUID					s_vertexRingBufferUId;
static SceUID					s_fragmentRingBufferUId;
static SceUID					s_fragmentUsseRingBufferUId;
static ClearVertex				*s_clearVertices			= NULL;
static uint16_t					*s_clearIndices				= NULL;
static SceGxmVertexProgram		*s_clearVertexProgram		= NULL;
static SceGxmFragmentProgram	*s_clearFragmentProgram		= NULL;
// !! Data added.
static SceGxmVertexProgram		*s_basicVertexProgram		= NULL;
static SceGxmFragmentProgram	*s_basicFragmentProgram		= NULL;
static BasicVertex				*s_basicVertices			= NULL;
static uint16_t					*s_basicIndices				= NULL;
static int32_t					s_basicVerticesUId;
static int32_t					s_basicIndiceUId;

//!! The program parameter for the transformation of the triangle
static Matrix4 s_finalTransformation; //
static const SceGxmProgramParameter *s_wvpParam = NULL;

/* Callback function to allocate memory for the shader patcher */
static void *patcherHostAlloc( void *userData, uint32_t size );

/* Callback function to allocate memory for the shader patcher */
static void patcherHostFree( void *userData, void *mem );

/*	Callback function for displaying a buffer */
static void displayCallback( const void *callbackData );

/*	Helper function to allocate memory and map it for the GPU */
static void *graphicsAlloc( SceKernelMemBlockType type, uint32_t size, uint32_t alignment, uint32_t attribs, SceUID *uid );

/*	Helper function to free memory mapped to the GPU */
static void graphicsFree( SceUID uid );

/* Helper function to allocate memory and map it as vertex USSE code for the GPU */
static void *vertexUsseAlloc( uint32_t size, SceUID *uid, uint32_t *usseOffset );

/* Helper function to free memory mapped as vertex USSE code for the GPU */
static void vertexUsseFree( SceUID uid );

/* Helper function to allocate memory and map it as fragment USSE code for the GPU */
static void *fragmentUsseAlloc( uint32_t size, SceUID *uid, uint32_t *usseOffset );

/* Helper function to free memory mapped as fragment USSE code for the GPU */
static void fragmentUsseFree( SceUID uid );

////////////////////// My functions and attributes ////////////////////////
static int currentCubeIndex = 0;

// Create one CubeSide
static void CreateCubeSide(BasicVertex* field, int cubeSide, float offset_X, float offset_Y, float offset_Z);

typedef std::vector <std::vector <BasicVertex*> >  vecBaseVertex;

static vecBaseVertex& getVecBaseVertex(char vec);

static void initializeRotation(char vecChar, int rotation_Axis);
static void initializeRotationTouch(char vecChar, int rotation_Axis, int cubeID);
static void rotateSide(char vecChar);
static void positionCubesAfterRotation(char vecChar, bool rotationDirection);

static bool isRotating = false;	// is a side currently rotating?

static bool isFrontTouchDisabled = false;
static bool isButtonControlDisabled = false;

static float currentRotationAngle = 0;
static bool rotationDirection;	// true   -->   clockwise
								// false  <--   counterClockwise
static int currentRotationAxis; // 0 = x
								// 1 = y
								// 2 = z
static std::vector< std::vector <Vector3> >	vec_tempRotatingCubes(9, std::vector<Vector3>(24));
static std::vector< std::vector <Vector3> >	vec_tempRotatingNormals(9, std::vector<Vector3>(24));

// 2D Vectors for *BasicVertex
// for each color_Side (6)																	// FACING THE RED SIDE:
static vecBaseVertex		vec_RedSide					(9, std::vector<BasicVertex*>(24)),	// Z_Front
							vec_YellowSide				(9, std::vector<BasicVertex*>(24)),	// X_Up
							vec_BlueSide				(9, std::vector<BasicVertex*>(24)),	// Y_Right
							vec_OrangeSide				(9, std::vector<BasicVertex*>(24)),	// Z_Back
							vec_WhiteSide				(9, std::vector<BasicVertex*>(24)),	// X_Down
							vec_GreenSide				(9, std::vector<BasicVertex*>(24)),	// Y_Left
// and each Axis (3)
							vec_X_Axis					(9, std::vector<BasicVertex*>(24)),	// X_Middle
							vec_Y_Axis					(9, std::vector<BasicVertex*>(24)),	// Y_Middle
							vec_Z_Axis					(9, std::vector<BasicVertex*>(24));	// Z_Middle
// current rotating side (1)
static char					vec_currentlyRotatingSide;	// currently rotating

// First touched coordinates for frontTouch
static Vector2 s_firstTouch_FRONT;

// Vectors for 6 normal_Vector3 which facing out (only for calculating raycast)
static std::vector <Vector3> vec_normals;

// The program parameter for the ROTATION of an Axis
static Matrix4 s_finalRotation;

// Quaternion
static Quat rotationQuat(0.0f, 0.0f, 1.0f, 0.0f);

/*	@brief Main entry point for the application
	@return Error code result of processing during execution: <c> SCE_OK </c> on success,
	or another code depending upon the error
*/
int main( void );

// !! Here we create the matrix.
void Update(void);

/*	@brief Initializes the graphics services and the libgxm graphics library
	@return Error code result of processing during execution: <c> SCE_OK </c> on success,
	or another code depending upon the error
*/
static int initGxm( void );

/*	 @brief Creates scenes with libgxm */
static void createGxmData( void );

/*	@brief Main rendering function to draw graphics to the display */
static void render( void );

/*	@brief render libgxm scenes */
static void renderGxm( void );

/*	@brief cycle display buffer */
static void cycleDisplayBuffers( void );

/*	@brief Destroy scenes with libgxm */
static void destroyGxmData( void );

/*	@brief Function to shut down libgxm and the graphics display services
	@return Error code result of processing during execution: <c> SCE_OK </c> on success,
	or another code depending upon the error
*/
static int shutdownGxm( void );

/*	@brief User main thread parameters */
extern const char			sceUserMainThreadName[]		= "simple_main_thr";
extern const int			sceUserMainThreadPriority	= SCE_KERNEL_DEFAULT_PRIORITY_USER;
extern const unsigned int	sceUserMainThreadStackSize	= SCE_KERNEL_STACK_SIZE_DEFAULT_USER_MAIN;

/*	@brief libc parameters */
unsigned int	sceLibcHeapSize	= 1*1024*1024;

/* Main entry point of program */
int main( void )
{
	int returnCode = SCE_OK;

	/* initialize libdbgfont and libgxm */
	returnCode =initGxm();
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

    SceDbgFontConfig config;
	memset( &config, 0, sizeof(SceDbgFontConfig) );
	config.fontSize = SCE_DBGFONT_FONTSIZE_LARGE;

	returnCode = sceDbgFontInit( &config );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* Message for SDK sample auto test */
	printf( "## simple: INIT SUCCEEDED ##\n" );

	/* create gxm graphics data */
	createGxmData();

     // Set sampling mode for input device.
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_DIGITALANALOG_WIDE);

	// enable touch_FRONT
	sceTouchSetSamplingState(SCE_TOUCH_PORT_FRONT, SCE_TOUCH_SAMPLING_STATE_START);

	// enable touch_BACK
	sceTouchSetSamplingState(SCE_TOUCH_PORT_BACK, SCE_TOUCH_SAMPLING_STATE_START);

	/* 6. main loop */
	while ( true)
	{
        Update();
		render();
		cycleDisplayBuffers();
	}

	// 10. wait until rendering is done 
	sceGxmFinish( s_context );

	// destroy gxm graphics data 
	destroyGxmData();

	// shutdown libdbgfont and libgxm
	returnCode = shutdownGxm();
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	// Message for SDK sample auto test
	printf( "## api_libdbgfont/simple: FINISHED ##\n" );

	return returnCode;
}

float makeFloat(unsigned char input)
{
    return (((float)(input)) / 255.0f * 2.0f) - 1.0f;
}

static float s_accumulatedTurningAngleX;
static float s_accumulatedTurningAngleY;

static float s_positionalDataTouch_FRONT_X;
static float s_positionalDataTouch_FRONT_Y;
static bool s_isTouched_FRONT = false;

static char touchedSide_char;
static int touchedCubeSideID;
static int touchedCubeID;
static char touchedColorID;

static float s_positionalDataTouch_BACK_X;
static float s_positionalDataTouch_BACK_Y;
static bool s_isTouched_BACK = false;

static float difference_X_vector_FRONT = 0.0f;
static float difference_Y_vector_FRONT = 0.0f;

void Update (void)
{	
	////////////////////////////		ROTATION		////////////////////////////
	// A side is currently rotating
	if(isRotating)
	{
		// rotate a side and increase the angle by 2 or -2
		rotateSide(vec_currentlyRotatingSide);

		if(rotationDirection)
		{
			currentRotationAngle += 2;
		}
		else
		{
			currentRotationAngle -= 2;
		}

		// if rotation is complete (90°)
		if(currentRotationAngle < -90 || currentRotationAngle > 90) // end rotation
		{
			// reset the angle and change cubes in vectors
			currentRotationAngle = 0;
			positionCubesAfterRotation(vec_currentlyRotatingSide, rotationDirection);
			isRotating = false;

			// Re-Enable frontTouch or buttonControl
			isFrontTouchDisabled = false;
			isButtonControlDisabled = false;
			s_isTouched_FRONT = false;

			// Return to default rotation
			rotationDirection = false;

			// Set first_Touch_Front to 0
			difference_X_vector_FRONT = 0.0f;
			difference_Y_vector_FRONT = 0.0f;
		}
	}

	////////////////////////////   LEFT_STICK CONTROL	////////////////////////////
	// read input_Data (control)
    SceCtrlData controlData;
    sceCtrlReadBufferPositive(0, &controlData, 1);


	// Clamp and set X-Rotation
	if(controlData.lx > 132 || controlData.lx < 124)
	{
		s_accumulatedTurningAngleX = -makeFloat(controlData.lx) * 0.01f;
	}
	else
	{
		s_accumulatedTurningAngleX = 0;
	}

	// Clamp and set Y-Rotation
	if(controlData.ly > 132 || controlData.ly < 124)
	{
		s_accumulatedTurningAngleY = makeFloat(controlData.ly) * 0.01f;
	}
	else
	{
		s_accumulatedTurningAngleY = 0;
	}

	////////////////////////////	 TOUCH CONTROL FRONT		////////////////////////////
	// Init Touch_FRONT
	SceTouchData dataTouchFront;

	// Retrieve data from the front touch panel
	sceTouchRead(SCE_TOUCH_PORT_FRONT, &dataTouchFront, 1);

	if(!isFrontTouchDisabled)
	{
		if(dataTouchFront.reportNum > 0)
		{
			if(!s_isTouched_FRONT)
			{
				// Disable Button Control
				isButtonControlDisabled = true;

				// normalize device coordinates
				s_positionalDataTouch_FRONT_X =	dataTouchFront.report[0].x / 1919.f * 2.f - 1.f;
				s_positionalDataTouch_FRONT_Y = 1.f - dataTouchFront.report[0].y / 1087.f * 2.f;

				// 4D homogeneous clip coordinates
				Vector4 ray_clip_start(s_positionalDataTouch_FRONT_X, s_positionalDataTouch_FRONT_Y, 0.1f, 1.f);
				Vector4 ray_clip_end(s_positionalDataTouch_FRONT_X, s_positionalDataTouch_FRONT_Y, 0.9f, 1.f);

				// transform back to object space
				ray_clip_start = inverse(s_finalTransformation) * ray_clip_start;
				ray_clip_end = inverse(s_finalTransformation) * ray_clip_end;

				// normalize vectors
				ray_clip_start /= ray_clip_start.getW();
				ray_clip_end /= ray_clip_end.getW();

				// get startpoint and endpoint for the 3D_Ray
				Point3 rayStart(ray_clip_start.getXYZ());
				Point3 rayEnd(ray_clip_end.getXYZ());

				// create 3D_Ray
				sce::Geometry::Aos::Ray ray_FrontTouch(rayStart, rayEnd);

				// Find the touched side_vector
				// Iterate through all 6 side vectors and their 9 cubes
				for(int i = 0; i < 6; ++i)
				{
					vecBaseVertex currentTestedCubeSide(9, std::vector<BasicVertex*>(24));

					switch(i)
					{
					case 0:
						{
							currentTestedCubeSide = vec_RedSide;
							touchedSide_char = 'r';
							break;
						}
					case 1:
						{
							currentTestedCubeSide = vec_BlueSide;
							touchedSide_char = 'b';
							break;
						}
					case 2:
						{
							currentTestedCubeSide = vec_OrangeSide;
							touchedSide_char = 'o';
							break;
						}
					case 3:
						{
							currentTestedCubeSide = vec_GreenSide;
							touchedSide_char = 'g';
							break;
						}
					case 4:
						{
							currentTestedCubeSide = vec_WhiteSide;
							touchedSide_char = 'w';
							break;
						}
					case 5:
						{
							currentTestedCubeSide = vec_YellowSide;
							touchedSide_char = 'y';
							break;
						}
					default:
						{
							break;
						}
					}

					// Get normals for calculation
					Vector3 normal = vec_normals[i];

					// calculate cube origin from normals
					Vector3 origin(	vec_normals[i].getX() * 0.6f,
									vec_normals[i].getY() * 0.6f,
									vec_normals[i].getZ() * 0.6f);

					// Vector3 for the ray_origin and ray_direction
					Vector3 ray_Origin_Vec(ray_FrontTouch.getOrigin());
					Vector3 ray_Direction_Vec(ray_FrontTouch.getDirection());

					// calculate dotproduct <ray, normal>
					float frontHitFactor = dot(ray_Direction_Vec, normal);

					// if float is negative -> smaller angle 90° and facing to the screen
					if(frontHitFactor < 0)
					{
						// alpha of HNF
						float hesse_normal_alpha = dot(origin - ray_Origin_Vec, normal) / dot(ray_Direction_Vec, normal);
					
						Vector3 intersection_point(ray_FrontTouch.getPointOnRay(sce::Vectormath::Simd::floatInVec(hesse_normal_alpha)));

						// build local 2D_coordinate system u/v
						Vector3 u_local_face(normal.getZ(), normal.getX(), normal.getY());
						Vector3 v_local_face(normal.getY(), normal.getZ(), normal.getX());
					
						// transform intersection_point into local coordinate system u/v
						float local_intersection_X = dot(u_local_face, intersection_point - origin);
						float local_intersection_Y = dot(v_local_face, intersection_point - origin);

						// check for intersection of the big cube
						if(	local_intersection_X > -0.6f && local_intersection_X < 0.6f &&
							local_intersection_Y > -0.6f && local_intersection_Y < 0.6f)
						{
							// check which of the nine cubes is touched
							float local_x;
							float local_y;

							if(touchedSide_char == 'r' || touchedSide_char == 'w' || touchedSide_char == 'g')
							{
								local_x = 0.6f;
								local_y = 0.6f;
						
								// cube 1-9
								int localCube = 0;

								//iterate all nine cubes of one side
								for(int j = 0; j < 3; ++j)
								{
									for(int k = 0; k < 3; ++k)
									{
										// check if touch is in 0.4 x 0.4 cubeface
										if( local_intersection_X < local_x && local_intersection_X > (local_x - 0.4f) &&
											local_intersection_Y < local_y && local_intersection_Y > (local_y - 0.4f))
										{
											// First Touch
											s_firstTouch_FRONT = Vector2(local_intersection_X, local_intersection_Y);

											// no first Touch possible again
											s_isTouched_FRONT = true;

											touchedCubeID = localCube;
											touchedColorID = touchedSide_char;
											touchedCubeSideID = i;
										}
										localCube++;

										if(touchedSide_char == 'w')
										{
											local_y -= 0.4f;
										}
										if(touchedSide_char == 'r' || touchedSide_char == 'g')
										{
											local_x -= 0.4f;
										}
									}
									if(touchedSide_char == 'w')
									{
										local_y = 0.6f;
										local_x -= 0.4f;
									}
									if(touchedSide_char == 'r' || touchedSide_char == 'g')
									{
										local_x = 0.6f;
										local_y -= 0.4f;
									}
								}
							}
							if(touchedSide_char == 'b' || touchedSide_char == 'o' || touchedSide_char == 'y')
							{
								local_x = -0.6f;
								local_y = -0.6f;

								// cube 0-8
								float localCube = 0;

								//iterate all nine cubes of one side
								for(int j = 0; j < 3; ++j)
								{
									for(int k = 0; k < 3; ++k)
									{
										// check if touch is in 0.4 x 0.4 cubeface
										if( local_intersection_X > local_x && local_intersection_X < (local_x + 0.4f) &&
											local_intersection_Y > local_y && local_intersection_Y < (local_y + 0.4f))
										{
											// First Touch
											s_firstTouch_FRONT = Vector2(local_intersection_X, local_intersection_Y);

											// no first Touch possible again
											s_isTouched_FRONT = true;

											touchedCubeID = localCube;
											touchedColorID = touchedSide_char;
											touchedCubeSideID = i;
										}
										localCube++;

										if(touchedSide_char == 'y')
										{
											local_y += 0.4f;
										}
										if(touchedSide_char == 'b' || touchedSide_char == 'o')
										{
											local_x += 0.4f;
										}
									}
									if(touchedSide_char == 'y')
									{
										local_y = -0.6f;
										local_x += 0.4f;
									}
									if(touchedSide_char == 'b' || touchedSide_char == 'o')
									{
										local_x = -0.6f;
										local_y += 0.4f;
									}
								}
							}
						}
					}
				}
			}


			// If a first touch is found
			if(s_isTouched_FRONT)
			{
				// normalize device coordinates
				s_positionalDataTouch_FRONT_X =	dataTouchFront.report[0].x / 1919.f * 2.f - 1.f;
				s_positionalDataTouch_FRONT_Y = 1.f - dataTouchFront.report[0].y / 1087.f * 2.f;

				// 4D homogeneous clip coordinates
				Vector4 ray_clip_start(s_positionalDataTouch_FRONT_X, s_positionalDataTouch_FRONT_Y, 0.1f, 1.f);
				Vector4 ray_clip_end(s_positionalDataTouch_FRONT_X, s_positionalDataTouch_FRONT_Y, 0.9f, 1.f);

				// transform back to object space
				ray_clip_start = inverse(s_finalTransformation) * ray_clip_start;
				ray_clip_end = inverse(s_finalTransformation) * ray_clip_end;

				// normalize vectors
				ray_clip_start /= ray_clip_start.getW();
				ray_clip_end /= ray_clip_end.getW();

				// get startpoint and endpoint for the 3D_Ray
				Point3 rayStart(ray_clip_start.getXYZ());
				Point3 rayEnd(ray_clip_end.getXYZ());

				// create 3D_Ray
				sce::Geometry::Aos::Ray ray_FrontTouch(rayStart, rayEnd);

				// Get normals for calculation
				Vector3 normal = vec_normals[touchedCubeSideID];

				// calculate cube origin from normals
				Vector3 origin(	vec_normals[touchedCubeSideID].getX() * 0.6f,
								vec_normals[touchedCubeSideID].getY() * 0.6f,
								vec_normals[touchedCubeSideID].getZ() * 0.6f);

				// Vector3 for the ray_origin and ray_direction
				Vector3 ray_Origin_Vec(ray_FrontTouch.getOrigin());
				Vector3 ray_Direction_Vec(ray_FrontTouch.getDirection());

				// calculate dotproduct <ray, normal>
				float frontHitFactor = dot(ray_Direction_Vec, normal);

				// if float is negative -> smaller angle 90° and facing to the screen
				if(frontHitFactor < 0)
				{
					// alpha of HNF
					float hesse_normal_alpha = dot(origin - ray_Origin_Vec, normal) / dot(ray_Direction_Vec, normal);
					
					Vector3 intersection_point(ray_FrontTouch.getPointOnRay(sce::Vectormath::Simd::floatInVec(hesse_normal_alpha)));

					// build local 2D_coordinate system u/v
					Vector3 u_local_face(normal.getZ(), normal.getX(), normal.getY());
					Vector3 v_local_face(normal.getY(), normal.getZ(), normal.getX());
					
					// transform intersection_point into local coordinate system u/v
					float local_intersection_X = dot(u_local_face, intersection_point - origin);
					float local_intersection_Y = dot(v_local_face, intersection_point - origin);

					difference_X_vector_FRONT = local_intersection_X - (float)s_firstTouch_FRONT.getX();
					difference_Y_vector_FRONT = local_intersection_Y - (float)s_firstTouch_FRONT.getY();

					std::cout << difference_X_vector_FRONT << std::endl;
					std::cout << difference_Y_vector_FRONT << std::endl;
					std::cout <<"____________________" << std::endl;
					std::cout << touchedColorID << std::endl;
					std::cout << touchedCubeID << std::endl;
					std::cout <<"____________________" << std::endl;

 					if(abs(difference_X_vector_FRONT) > 0.25f && abs(difference_X_vector_FRONT) > abs(difference_Y_vector_FRONT))
 					{
						if(difference_X_vector_FRONT > 0)
						{
							rotationDirection = !rotationDirection;
						}
 						initializeRotationTouch(touchedColorID, 1, touchedCubeID);
 						isFrontTouchDisabled = true;
 					}
 					else if(abs(difference_Y_vector_FRONT) > 0.25f && abs(difference_Y_vector_FRONT) > abs(difference_X_vector_FRONT))
 					{
						if(difference_Y_vector_FRONT < 0)
						{
							rotationDirection = !rotationDirection;
						}
 						initializeRotationTouch(touchedColorID, 0, touchedCubeID);
 						isFrontTouchDisabled = true;
 					}
				}
			}
		}
	}



	////////////////////////////	 TOUCH CONTROL BACK		////////////////////////////
	// Init Touch_BACK
	SceTouchData dataTouchBack;

	// Retrieve data from the rear touch panel
	sceTouchRead(SCE_TOUCH_PORT_BACK, &dataTouchBack, 1);

	float differencePosition_BACK_X = 0.0f;
	float differencePosition_BACK_Y = 0.0f;

	if(dataTouchBack.reportNum > 0)
	{	
		if(!s_isTouched_BACK)
		{
			// normalize device coordinates
			s_positionalDataTouch_BACK_X =	dataTouchBack.report[0].x / 1919.f * 2.f - 1.f;
			s_positionalDataTouch_BACK_Y = 1.f - dataTouchBack.report[0].y / 889.f * 2.f;
			s_isTouched_BACK = true;
		}
		else
		{
			// normalize device coordinates
			differencePosition_BACK_X =	dataTouchBack.report[0].x / 1919.f * 2.f - 1.f;
			differencePosition_BACK_Y = 1.f - dataTouchBack.report[0].y / 889.f * 2.f;

			differencePosition_BACK_X -= s_positionalDataTouch_BACK_X;
			differencePosition_BACK_Y -= s_positionalDataTouch_BACK_Y;
		}
	}
	else
	{
		s_isTouched_BACK = false;
	}

	if(s_isTouched_BACK)
	{
		s_accumulatedTurningAngleX = differencePosition_BACK_X * 0.02f;
		s_accumulatedTurningAngleY = differencePosition_BACK_Y * 0.02f;
	}
	
	////////////////////////////	 BUTTON CONTROL		////////////////////////////

	if(!isButtonControlDisabled)
	{
		// Rotation_Direction Control
		if((controlData.buttons & SCE_CTRL_L) != 0)
		{
			rotationDirection = false;
		}
		else if((controlData.buttons & SCE_CTRL_R) != 0)
		{
			rotationDirection = true;
		}

		// Z AXIS CONTROL
		// SQUARE_BUTTON pressed
		if((controlData.buttons & SCE_CTRL_SQUARE) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('r', 2);
		}
		// TRIANGLE_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_TRIANGLE) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('z', 2);
		}
		// CIRCLE_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_CIRCLE) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('o', 2);
		}

		// Y AXIS CONTROL
		// CROSS_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_CROSS) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('g', 0);
		}
		// LEFT_BUTTON_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_LEFT) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('u', 0); // 'u' == Y_Axis
		}
		// UP_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_UP) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('b', 0);
		}

		// X AXIS CONTROL
		// RIGHT_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_RIGHT) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('w', 1);
		}
		// DOWN_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_DOWN) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('x', 1);
		}
		// SELECT_BUTTON pressed
		else if((controlData.buttons & SCE_CTRL_SELECT) != 0 && !isRotating)
		{
			// Disable FrontTouch
			isFrontTouchDisabled = true;

			initializeRotation('y', 1);
		}
	}

	////////////////////////////	   ROTATION			////////////////////////////
	// Use Quaternion for rotation
	Quat rotationVelocity(s_accumulatedTurningAngleY, s_accumulatedTurningAngleX, 0.0f, 0.0f);

	rotationQuat += 2.5 * rotationVelocity * rotationQuat;
	rotationQuat = normalize(rotationQuat);

	// calculate the different matrices for the final transformation of objects
    Matrix4 rotation = Matrix4(rotationQuat, Vector3(0.0f, 0.0f, 0.0f));

    Matrix4 lookAt = Matrix4::lookAt(Point3(0.0f, 0.0f, -3.0f), Point3(0.0f, 0.0f, 0.0f), Vector3(0.0f, -1.0f, 0.0f));

    Matrix4 perspective = Matrix4::perspective(	3.141592f / 4.0f,
												(float)DISPLAY_WIDTH/(float)DISPLAY_HEIGHT,
												0.1f,
												10.0f);
    
	// the final transformation
    s_finalTransformation = perspective * lookAt * rotation; 
};

/* Initialize libgxm */
int initGxm( void )
{
/* ---------------------------------------------------------------------
	2. Initialize libgxm

	First we must initialize the libgxm library by calling sceGxmInitialize.
	The single argument to this function is the size of the parameter buffer to
	allocate for the GPU.  We will use the default 16MiB here.

	Once initialized, we need to create a rendering context to allow to us
	to render scenes on the GPU.  We use the default initialization
	parameters here to set the sizes of the various context ring buffers.

	Finally we create a render target to describe the geometry of the back
	buffers we will render to.  This object is used purely to schedule
	rendering jobs for the given dimensions, the color surface and
	depth/stencil surface must be allocated separately.
	--------------------------------------------------------------------- */

	int returnCode = SCE_OK;

	/* set up parameters */
	SceGxmInitializeParams initializeParams;
	memset( &initializeParams, 0, sizeof(SceGxmInitializeParams) );
	initializeParams.flags = 0;
	initializeParams.displayQueueMaxPendingCount = DISPLAY_MAX_PENDING_SWAPS;
	initializeParams.displayQueueCallback = displayCallback;
	initializeParams.displayQueueCallbackDataSize = sizeof(DisplayData);
	initializeParams.parameterBufferSize = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE;

	/* start libgxm */
	returnCode = sceGxmInitialize( &initializeParams );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* allocate ring buffer memory using default sizes */
	void *vdmRingBuffer = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE, 4, SCE_GXM_MEMORY_ATTRIB_READ, &s_vdmRingBufferUId );

	void *vertexRingBuffer = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE, 4, SCE_GXM_MEMORY_ATTRIB_READ, &s_vertexRingBufferUId );

	void *fragmentRingBuffer = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE, 4, SCE_GXM_MEMORY_ATTRIB_READ, &s_fragmentRingBufferUId );

	uint32_t fragmentUsseRingBufferOffset;
	void *fragmentUsseRingBuffer = fragmentUsseAlloc( SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE, &s_fragmentUsseRingBufferUId, &fragmentUsseRingBufferOffset );

	/* create a rendering context */
	memset( &s_contextParams, 0, sizeof(SceGxmContextParams) );
	s_contextParams.hostMem = malloc( SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE );
	s_contextParams.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
	s_contextParams.vdmRingBufferMem = vdmRingBuffer;
	s_contextParams.vdmRingBufferMemSize = SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE;
	s_contextParams.vertexRingBufferMem = vertexRingBuffer;
	s_contextParams.vertexRingBufferMemSize = SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE;
	s_contextParams.fragmentRingBufferMem = fragmentRingBuffer;
	s_contextParams.fragmentRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE;
	s_contextParams.fragmentUsseRingBufferMem = fragmentUsseRingBuffer;
	s_contextParams.fragmentUsseRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE;
	s_contextParams.fragmentUsseRingBufferOffset = fragmentUsseRingBufferOffset;
	returnCode = sceGxmCreateContext( &s_contextParams, &s_context );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* set buffer sizes for this sample */
	const uint32_t patcherBufferSize = 64*1024;
	const uint32_t patcherVertexUsseSize = 64*1024;
	const uint32_t patcherFragmentUsseSize = 64*1024;

	/* allocate memory for buffers and USSE code */
	void *patcherBuffer = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, patcherBufferSize, 4, SCE_GXM_MEMORY_ATTRIB_WRITE|SCE_GXM_MEMORY_ATTRIB_WRITE, &s_patcherBufferUId );

	uint32_t patcherVertexUsseOffset;
	void *patcherVertexUsse = vertexUsseAlloc( patcherVertexUsseSize, &s_patcherVertexUsseUId, &patcherVertexUsseOffset );

	uint32_t patcherFragmentUsseOffset;
	void *patcherFragmentUsse = fragmentUsseAlloc( patcherFragmentUsseSize, &s_patcherFragmentUsseUId, &patcherFragmentUsseOffset );

	/* create a shader patcher */
	SceGxmShaderPatcherParams patcherParams;
	memset( &patcherParams, 0, sizeof(SceGxmShaderPatcherParams) );
	patcherParams.userData = NULL;
	patcherParams.hostAllocCallback = &patcherHostAlloc;
	patcherParams.hostFreeCallback = &patcherHostFree;
	patcherParams.bufferAllocCallback = NULL;
	patcherParams.bufferFreeCallback = NULL;
	patcherParams.bufferMem = patcherBuffer;
	patcherParams.bufferMemSize = patcherBufferSize;
	patcherParams.vertexUsseAllocCallback = NULL;
	patcherParams.vertexUsseFreeCallback = NULL;
	patcherParams.vertexUsseMem = patcherVertexUsse;
	patcherParams.vertexUsseMemSize = patcherVertexUsseSize;
	patcherParams.vertexUsseOffset = patcherVertexUsseOffset;
	patcherParams.fragmentUsseAllocCallback = NULL;
	patcherParams.fragmentUsseFreeCallback = NULL;
	patcherParams.fragmentUsseMem = patcherFragmentUsse;
	patcherParams.fragmentUsseMemSize = patcherFragmentUsseSize;
	patcherParams.fragmentUsseOffset = patcherFragmentUsseOffset;
	returnCode = sceGxmShaderPatcherCreate( &patcherParams, &s_shaderPatcher );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* create a render target */
	memset( &s_renderTargetParams, 0, sizeof(SceGxmRenderTargetParams) );
	s_renderTargetParams.flags = 0;
	s_renderTargetParams.width = DISPLAY_WIDTH;
	s_renderTargetParams.height = DISPLAY_HEIGHT;
	s_renderTargetParams.scenesPerFrame = 1;
	s_renderTargetParams.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	s_renderTargetParams.multisampleLocations	= 0;
	s_renderTargetParams.driverMemBlock = SCE_UID_INVALID_UID;

	returnCode = sceGxmCreateRenderTarget( &s_renderTargetParams, &s_renderTarget );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );


/* ---------------------------------------------------------------------
	3. Allocate display buffers, set up the display queue

	We will allocate our back buffers in CDRAM, and create a color
	surface for each of them.

	To allow display operations done by the CPU to be synchronized with
	rendering done by the GPU, we also create a SceGxmSyncObject for each
	display buffer.  This sync object will be used with each scene that
	renders to that buffer and when queueing display flips that involve
	that buffer (either flipping from or to).

	Finally we create a display queue object that points to our callback
	function.
	--------------------------------------------------------------------- */

	/* allocate memory and sync objects for display buffers */
	for ( unsigned int i = 0 ; i < DISPLAY_BUFFER_COUNT ; ++i )
	{
		/* allocate memory with large size to ensure physical contiguity */
		s_displayBufferData[i] = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RWDATA, ALIGN(4*DISPLAY_STRIDE_IN_PIXELS*DISPLAY_HEIGHT, 1*1024*1024), SCE_GXM_COLOR_SURFACE_ALIGNMENT, SCE_GXM_MEMORY_ATTRIB_READ|SCE_GXM_MEMORY_ATTRIB_WRITE, &s_displayBufferUId[i] );
		SCE_DBG_ALWAYS_ASSERT( s_displayBufferData[i] );

		/* memset the buffer to debug color */
		for ( unsigned int y = 0 ; y < DISPLAY_HEIGHT ; ++y )
		{
			unsigned int *row = (unsigned int *)s_displayBufferData[i] + y*DISPLAY_STRIDE_IN_PIXELS;

			for ( unsigned int x = 0 ; x < DISPLAY_WIDTH ; ++x )
			{
				row[x] = 0x0;
			}
		}

		/* initialize a color surface for this display buffer */
		returnCode = sceGxmColorSurfaceInit( &s_displaySurface[i], DISPLAY_COLOR_FORMAT, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
											 SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_STRIDE_IN_PIXELS, s_displayBufferData[i] );
		SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

		/* create a sync object that we will associate with this buffer */
		returnCode = sceGxmSyncObjectCreate( &s_displayBufferSync[i] );
		SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );
	}

	/* compute the memory footprint of the depth buffer */
	const uint32_t alignedWidth = ALIGN( DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX );
	const uint32_t alignedHeight = ALIGN( DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY );
	uint32_t sampleCount = alignedWidth*alignedHeight;
	uint32_t depthStrideInSamples = alignedWidth;

	/* allocate it */
	void *depthBufferData = graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, 4*sampleCount, SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT, SCE_GXM_MEMORY_ATTRIB_READ|SCE_GXM_MEMORY_ATTRIB_WRITE, &s_depthBufferUId );

	/* create the SceGxmDepthStencilSurface structure */
	returnCode = sceGxmDepthStencilSurfaceInit( &s_depthSurface, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24, SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, depthStrideInSamples, depthBufferData, NULL );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	return returnCode;
}

// create a cube with 6 different sides
void CreateOneCube(int cubeCount, int cubeGroupYCount, int cubeGroupZIndex, int verticesCount, int indicesCount, int indicesBaseCount, float offset_X, float offset_Y, float offset_Z)
{
	// The vertices
    for(int cubeSide = 1; cubeSide <= 6; ++cubeSide)
	{
        CreateCubeSide(&(s_basicVertices[verticesCount]), cubeSide, offset_X, offset_Y, offset_Z);
        verticesCount += 4;
    }

    // The indices.
    for(int side = 0; side < 6; ++side)
    {
        int baseIndex = side * 4 + indicesBaseCount;
        s_basicIndices[indicesCount++] = baseIndex;
        s_basicIndices[indicesCount++] = baseIndex + 1;
        s_basicIndices[indicesCount++] = baseIndex + 2;

        s_basicIndices[indicesCount++] = baseIndex;
        s_basicIndices[indicesCount++] = baseIndex + 3;
        s_basicIndices[indicesCount++] = baseIndex + 2;
    }

	int verticesStartIndexCube = cubeCount * 4*6;

	int vectorIndex = 0;

	// Group Red Side (Cubes [0] - [8])
	if(cubeCount < 9)
	{
		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_RedSide[cubeCount][vectorIndex++] = &s_basicVertices[i];
		}
	}

	// Group Z_Axis  (Cubes [9] - [17])
	if(cubeCount >= 9 && cubeCount < 18)
	{
		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_Z_Axis[cubeCount - 9][vectorIndex++] = &s_basicVertices[i];
		}
	}

	// Group Orange Side (Cubes [18] - [26])
	if(cubeCount >= 18)
	{
		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_OrangeSide[cubeCount - 18][vectorIndex++] = &s_basicVertices[i];
		}
	}
	
	// Group Green Side
	if(cubeCount % 3 == 0)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_GreenSide[cubeGroupYCount][vectorIndex++] = &s_basicVertices[i];
		}
	}
	
	// Group Y_Axis
	if(cubeCount % 3 == 1)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_Y_Axis[cubeGroupYCount][vectorIndex++] = &s_basicVertices[i];
		}
	}

	// Group Blue Side
	if(cubeCount % 3 == 2)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_BlueSide[cubeGroupYCount][vectorIndex++] = &s_basicVertices[i];
		}
	}

	int checkCubeVariable = cubeCount % 9;
	int indexVariableZ = cubeCount % 3;

	// Group White Side
	if(checkCubeVariable == 0 || checkCubeVariable == 1 || checkCubeVariable == 2)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_WhiteSide[indexVariableZ + (cubeGroupZIndex * 3)][vectorIndex++] = &s_basicVertices[i];
		}
	}
	
	// Group X_Axis
	if(checkCubeVariable == 3 || checkCubeVariable == 4 || checkCubeVariable == 5)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_X_Axis[indexVariableZ + (cubeGroupZIndex * 3)][vectorIndex++] = &s_basicVertices[i];
		}
	}

	// Group Yellow Side
	if(checkCubeVariable == 6 || checkCubeVariable == 7 || checkCubeVariable == 8)
	{
		vectorIndex = 0;

		for(int i = verticesStartIndexCube; i < verticesStartIndexCube + 4*6; ++i)
		{
			vec_YellowSide[indexVariableZ + (cubeGroupZIndex * 3)][vectorIndex++] = &s_basicVertices[i];
		}
	}
}

// create one specific cubeside (six different colors):
// 1 == red
// 2 == blue
// 3 == orange
// 4 == green
// 5 == white
// 6 == yellow
void CreateCubeSide(BasicVertex* field, int cubeSide, float offset_X, float offset_Y, float offset_Z)
{
	switch (cubeSide)
	{
	case 1: // RED
		{
			field[0].position[0] = -0.60f + offset_X;
			field[0].position[1] = -0.20f + offset_Y;
			field[0].position[2] = -0.60f + offset_Z;
			field[0].color = 0x000000FF;

			field[1].position[0] = -0.60f + offset_X;
			field[1].position[1] = -0.60f + offset_Y;
			field[1].position[2] = -0.60f + offset_Z;
			field[1].color = 0x000000FF;

			field[2].position[0] = -0.20f + offset_X;
			field[2].position[1] = -0.60f + offset_Y;
			field[2].position[2] = -0.60f + offset_Z;
			field[2].color = 0x000000FF;

			field[3].position[0] = -0.20f + offset_X;
			field[3].position[1] = -0.20f + offset_Y;
			field[3].position[2] = -0.60f + offset_Z;
			field[3].color = 0x000000FF;

			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 2)
					{
						field[i].normal[j] = -1.0f;
					}
				}
			}

			break;
		}
	case 2: // BLUE
		{
			field[0].position[0] = -0.20f + offset_X;
			field[0].position[1] = -0.20f + offset_Y;
			field[0].position[2] = -0.60f + offset_Z;
			field[0].color = 0x00FF0000;

			field[1].position[0] = -0.20f + offset_X;
			field[1].position[1] = -0.60f + offset_Y;
			field[1].position[2] = -0.60f + offset_Z;
			field[1].color = 0x00FF0000;

			field[2].position[0] = -0.20f + offset_X;
			field[2].position[1] = -0.60f + offset_Y;
			field[2].position[2] = -0.20f + offset_Z;
			field[2].color = 0x00FF0000;

			field[3].position[0] = -0.20f + offset_X;
			field[3].position[1] = -0.20f + offset_Y;
			field[3].position[2] = -0.20f + offset_Z;
			field[3].color = 0x00FF0000;
			
			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 0)
					{
						field[i].normal[j] = 1.0f;
					}
				}
			}

			break;
		}
	case 3: // ORANGE
		{
			field[0].position[0] = -0.20f + offset_X;
			field[0].position[1] = -0.20f + offset_Y;
			field[0].position[2] = -0.20f + offset_Z;
			field[0].color = 0x000088FF;

			field[1].position[0] = -0.20f + offset_X;
			field[1].position[1] = -0.60f + offset_Y;
			field[1].position[2] = -0.20f + offset_Z;
			field[1].color = 0x000088FF;

			field[2].position[0] = -0.60f + offset_X;
			field[2].position[1] = -0.60f + offset_Y;
			field[2].position[2] = -0.20f + offset_Z;
			field[2].color = 0x000088FF;

			field[3].position[0] = -0.60f + offset_X;
			field[3].position[1] = -0.20f + offset_Y;
			field[3].position[2] = -0.20f + offset_Z;
			field[3].color = 0x000088FF;
			
			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 2)
					{
						field[i].normal[j] = 1.0f;
					}
				}
			}

			break;
		}
	case 4: // GREEN
		{
			field[0].position[0] = -0.60f + offset_X;
			field[0].position[1] = -0.20f + offset_Y;
			field[0].position[2] = -0.20f + offset_Z;
			field[0].color = 0x0000FF00;

			field[1].position[0] = -0.60f + offset_X;
			field[1].position[1] = -0.60f + offset_Y;
			field[1].position[2] = -0.20f + offset_Z;
			field[1].color = 0x0000FF00;

			field[2].position[0] = -0.60f + offset_X;
			field[2].position[1] = -0.60f + offset_Y;
			field[2].position[2] = -0.60f + offset_Z;
			field[2].color = 0x0000FF00;

			field[3].position[0] = -0.60f + offset_X;
			field[3].position[1] = -0.20f + offset_Y;
			field[3].position[2] = -0.60f + offset_Z;
			field[3].color = 0x0000FF00;
			
			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 0)
					{
						field[i].normal[j] = -1.0f;
					}
				}
			}

			break;
		}
	case 5: // WHITE
		{
			field[0].position[0] = -0.60f + offset_X;
			field[0].position[1] = -0.60f + offset_Y;
			field[0].position[2] = -0.60f + offset_Z;
			field[0].color = 0x00FFFFFF;

			field[1].position[0] = -0.60f + offset_X;
			field[1].position[1] = -0.60f + offset_Y;
			field[1].position[2] = -0.20f + offset_Z;
			field[1].color = 0x00FFFFFF;

			field[2].position[0] = -0.20f + offset_X;
			field[2].position[1] = -0.60f + offset_Y;
			field[2].position[2] = -0.20f + offset_Z;
			field[2].color = 0x00FFFFFF;

			field[3].position[0] = -0.20f + offset_X;
			field[3].position[1] = -0.60f + offset_Y;
			field[3].position[2] = -0.60f + offset_Z;
			field[3].color = 0x00FFFFFF;
			
			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 1)
					{
						field[i].normal[j] = -1.0f;
					}
				}
			}

			break;
		}
	case 6: // YELLOW
		{
			field[0].position[0] = -0.60f + offset_X;
			field[0].position[1] = -0.20f + offset_Y;
			field[0].position[2] = -0.60f + offset_Z;
			field[0].color = 0x0000FFFF;

			field[1].position[0] = -0.60f + offset_X;
			field[1].position[1] = -0.20f + offset_Y;
			field[1].position[2] = -0.20f + offset_Z;
			field[1].color = 0x0000FFFF;

			field[2].position[0] = -0.20f + offset_X;
			field[2].position[1] = -0.20f + offset_Y;
			field[2].position[2] = -0.20f + offset_Z;
			field[2].color = 0x0000FFFF;

			field[3].position[0] = -0.20f + offset_X;
			field[3].position[1] = -0.20f + offset_Y;
			field[3].position[2] = -0.60f + offset_Z;
			field[3].color = 0x0000FFFF;
			
			// init normals
			for(int i = 0; i < 4; ++i)
			{
				for(int j = 0; j < 3; ++j)
				{
					field[i].normal[j] = 0.0f;

					if(j == 1)
					{
						field[i].normal[j] = 1.0f;
					}
				}
			}

			break;
		}
	default:
		{
			break;
		}
	}

	// init local u/v for dark borders in shader
	field[0].uv[0] = 0.0f;
	field[0].uv[1] = 0.0f;

	field[1].uv[0] = 0.0f;
	field[1].uv[1] = 1.0f;

	field[2].uv[0] = 1.0f;
	field[2].uv[1] = 1.0f;

	field[3].uv[0] = 0.0f;
	field[3].uv[1] = 1.0f;
}

///////// Functions for Rotation /////////
//										//
void initializeRotation(char vecChar, int rotation_Axis)
{
	for(int i = 0; i < 9; ++i)
	{
		for(int j = 0; j < 4*6; ++j)
		{
			vec_currentlyRotatingSide = vecChar;

			// save current coordinates for Position
			vec_tempRotatingCubes[i][j] = Vector3(	getVecBaseVertex(vecChar)[i][j]->position[0], 
													getVecBaseVertex(vecChar)[i][j]->position[1],
													getVecBaseVertex(vecChar)[i][j]->position[2]);

			// save current coordinates for Normal
			vec_tempRotatingNormals[i][j] = Vector3(getVecBaseVertex(vecChar)[i][j]->normal[0], 
													getVecBaseVertex(vecChar)[i][j]->normal[1],
													getVecBaseVertex(vecChar)[i][j]->normal[2]);
		}
	}

	// set the current rotation Axis
	currentRotationAxis = rotation_Axis;
	isRotating = true;
}


/*
vecChar = Touched colorSide
rotation_Axis:
0 = Y-Axis
1 = X-Axis
*/
void initializeRotationTouch(char vecChar, int rotation_Axis, int cubeID)
{
	switch(vecChar)
	{
	case 'r':
		{
			switch(rotation_Axis)
			{
			case 1: // X_Axis
				{
					if(		cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('y', 1);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('x', 1);
					}
					else if(cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('w', 1);
					}
					break;
				}
			case 0: // Y_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('b', 0);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('u', 0);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('g', 0);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}





	case 'b':
		{
			switch(rotation_Axis)
			{
			case 0: // X_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('y', 1);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('x', 1);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('w', 1);
					}
					break;
				}
			case 1: // Y_Axis
				{
					if(		cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('o', 2);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('z', 2);
					}
					else if(cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('r', 2);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}





	case 'o':
		{
			switch(rotation_Axis)
			{
			case 1: // X_Axis
				{
					if(		cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('y', 1);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('x', 1);
					}
					else if(cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('w', 1);
					}
					break;
				}
			case 0: // Y_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('b', 0);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('u', 0);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('g', 0);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}





	case 'g':
		{
			switch(rotation_Axis)
			{
			case 0: // X_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('y', 1);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('x', 1);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('w', 1);
					}
					break;
				}
			case 1: // Y_Axis
				{
					if(		cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('r', 2);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('z', 2);
					}
					else if(cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('o', 2);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}




	case 'w':
		{
			switch(rotation_Axis)
			{
			case 0: // local_X_Axis
				{
					if(		cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('r', 2);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('z', 2);
					}
					else if(cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('o', 2);
					}
					break;
				}
			case 1: // local_Y_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('b', 0);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('u', 0);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('g', 0);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}






	case 'y':
		{
			switch(rotation_Axis)
			{
			case 0: // X_Axis
				{
					if(		cubeID == 2 || cubeID == 1 || cubeID == 0)
					{
						initializeRotation('r', 2);
					}
					else if(cubeID == 5 || cubeID == 4 || cubeID == 3)
					{
						initializeRotation('z', 2);
					}
					else if(cubeID == 8 || cubeID == 7 || cubeID == 6)
					{
						initializeRotation('o', 2);
					}
					break;
				}
			case 1: // Y_Axis
				{
					if(		cubeID == 8 || cubeID == 5 || cubeID == 2)
					{
						initializeRotation('b', 0);
					}
					else if(cubeID == 7 || cubeID == 4 || cubeID == 1)
					{
						initializeRotation('u', 0);
					}
					else if(cubeID == 6 || cubeID == 3 || cubeID == 0)
					{
						initializeRotation('g', 0);
					}
					break;
				}
			default:
				{
					break;
				}
			}
			break;
		}
	default:
		{
			break;
		}
	}
}

void rotateSide(char vecChar)
{	
	// calculate new rotation matrices
	if(currentRotationAxis == 0)
	{
		s_finalRotation = Matrix4::rotationX(currentRotationAngle * RAD_ROTATION);
	}
	else if(currentRotationAxis == 1)
	{
		s_finalRotation = Matrix4::rotationY(currentRotationAngle * RAD_ROTATION);
	}
	else if(currentRotationAxis == 2)
	{
		s_finalRotation = Matrix4::rotationZ(currentRotationAngle * RAD_ROTATION);
	}

	// 
	for(int i = 0; i < 9; ++i)
	{
		for(int j = 0; j < 4*6; ++j)
		{
			// calculate new coordinates for Position
			Vector4 newRotationCoords = s_finalRotation * vec_tempRotatingCubes[i][j];

			getVecBaseVertex(vecChar)[i][j]->position[0] = newRotationCoords.getX();
			getVecBaseVertex(vecChar)[i][j]->position[1] = newRotationCoords.getY();
			getVecBaseVertex(vecChar)[i][j]->position[2] = newRotationCoords.getZ();

			// calculate new coordinates for Normal
			Vector4 newRotationCoordsNormals = s_finalRotation * vec_tempRotatingNormals[i][j];

			getVecBaseVertex(vecChar)[i][j]->normal[0] = newRotationCoordsNormals.getX();
			getVecBaseVertex(vecChar)[i][j]->normal[1] = newRotationCoordsNormals.getY();
			getVecBaseVertex(vecChar)[i][j]->normal[2] = newRotationCoordsNormals.getZ();
		}
	}
}

// get the current rotating vector with the nine cubes
vecBaseVertex& getVecBaseVertex(char vec)
{
	switch(vec)
	{
	case 'r':
		{
			return vec_RedSide;
		}
	case 'y':
		{
			return vec_YellowSide;
		}
	case 'b':
		{
			return vec_BlueSide;
		}
	case 'o':
		{
			return vec_OrangeSide;
		}
	case 'w':
		{
			return vec_WhiteSide;
		}
	case 'g':
		{
			return vec_GreenSide;
		}
	case 'x':
		{
			return vec_X_Axis;
		}
	case 'u':
		{
			return vec_Y_Axis;
		}
	case 'z':
		{
			return vec_Z_Axis;
		}
	default:
		{
			return vec_RedSide;
		}
	}
}

// after rotation is complete, relocate the specific cubes in the vectors
void positionCubesAfterRotation(char vecChar, bool rotationDirection)
{
	switch(vecChar)
	{
	case 'r':
		{
			vecBaseVertex tmp_vec = vec_RedSide;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_RedSide[0] = tmp_vec[6];
				vec_RedSide[1] = tmp_vec[3];
				vec_RedSide[2] = tmp_vec[0];
				vec_RedSide[3] = tmp_vec[7];
				vec_RedSide[4] = tmp_vec[4];
				vec_RedSide[5] = tmp_vec[1];
				vec_RedSide[6] = tmp_vec[8];
				vec_RedSide[7] = tmp_vec[5];
				vec_RedSide[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_RedSide[0] = tmp_vec[2];
				vec_RedSide[1] = tmp_vec[5];
				vec_RedSide[2] = tmp_vec[8];
				vec_RedSide[3] = tmp_vec[1];
				vec_RedSide[4] = tmp_vec[4];
				vec_RedSide[5] = tmp_vec[7];
				vec_RedSide[6] = tmp_vec[0];
				vec_RedSide[7] = tmp_vec[3];
				vec_RedSide[8] = tmp_vec[6];
			}

			// Around RedSide
			vec_WhiteSide[0] = vec_RedSide[0];
			vec_WhiteSide[1] = vec_RedSide[1];
			vec_WhiteSide[2] = vec_RedSide[2];

			vec_BlueSide[0] = vec_RedSide[2];
			vec_BlueSide[1] = vec_RedSide[5];
			vec_BlueSide[2] = vec_RedSide[8];

			vec_YellowSide[2] = vec_RedSide[8];
			vec_YellowSide[1] = vec_RedSide[7];
			vec_YellowSide[0] = vec_RedSide[6];

			vec_GreenSide[2] = vec_RedSide[6];
			vec_GreenSide[1] = vec_RedSide[3];
			vec_GreenSide[0] = vec_RedSide[0];
			
			// Switching Axis
			vec_X_Axis[0] = vec_RedSide[3];
			vec_X_Axis[2] = vec_RedSide[5];

			vec_Y_Axis[0] = vec_RedSide[1];
			vec_Y_Axis[2] = vec_RedSide[7];
			break;
		}
		case 'z':
		{
			vecBaseVertex tmp_vec = vec_Z_Axis;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_Z_Axis[0] = tmp_vec[6];
				vec_Z_Axis[1] = tmp_vec[3];
				vec_Z_Axis[2] = tmp_vec[0];
				vec_Z_Axis[3] = tmp_vec[7];
				vec_Z_Axis[4] = tmp_vec[4];
				vec_Z_Axis[5] = tmp_vec[1];
				vec_Z_Axis[6] = tmp_vec[8];
				vec_Z_Axis[7] = tmp_vec[5];
				vec_Z_Axis[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_Z_Axis[0] = tmp_vec[2];
				vec_Z_Axis[1] = tmp_vec[5];
				vec_Z_Axis[2] = tmp_vec[8];
				vec_Z_Axis[3] = tmp_vec[1];
				vec_Z_Axis[4] = tmp_vec[4];
				vec_Z_Axis[5] = tmp_vec[7];
				vec_Z_Axis[6] = tmp_vec[0];
				vec_Z_Axis[7] = tmp_vec[3];
				vec_Z_Axis[8] = tmp_vec[6];
			}

			// Around RedSide
			vec_WhiteSide[3] = vec_Z_Axis[0];
			vec_WhiteSide[4] = vec_Z_Axis[1];
			vec_WhiteSide[5] = vec_Z_Axis[2];

			vec_BlueSide[3] = vec_Z_Axis[2];
			vec_BlueSide[4] = vec_Z_Axis[5];
			vec_BlueSide[5] = vec_Z_Axis[8];

			vec_YellowSide[5] = vec_Z_Axis[8];
			vec_YellowSide[4] = vec_Z_Axis[7];
			vec_YellowSide[3] = vec_Z_Axis[6];

			vec_GreenSide[5] = vec_Z_Axis[6];
			vec_GreenSide[4] = vec_Z_Axis[3];
			vec_GreenSide[3] = vec_Z_Axis[0];
			
			// Switching Axis
			vec_X_Axis[3] = vec_Z_Axis[3];
			vec_X_Axis[5] = vec_Z_Axis[5];

			vec_Y_Axis[3] = vec_Z_Axis[1];
			vec_Y_Axis[5] = vec_Z_Axis[7];
			break;
		}
	case 'o':
		{
			vecBaseVertex tmp_vec = vec_OrangeSide;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_OrangeSide[0] = tmp_vec[6];
				vec_OrangeSide[1] = tmp_vec[3];
				vec_OrangeSide[2] = tmp_vec[0];
				vec_OrangeSide[3] = tmp_vec[7];
				vec_OrangeSide[4] = tmp_vec[4];
				vec_OrangeSide[5] = tmp_vec[1];
				vec_OrangeSide[6] = tmp_vec[8];
				vec_OrangeSide[7] = tmp_vec[5];
				vec_OrangeSide[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_OrangeSide[0] = tmp_vec[2];
				vec_OrangeSide[1] = tmp_vec[5];
				vec_OrangeSide[2] = tmp_vec[8];
				vec_OrangeSide[3] = tmp_vec[1];
				vec_OrangeSide[4] = tmp_vec[4];
				vec_OrangeSide[5] = tmp_vec[7];
				vec_OrangeSide[6] = tmp_vec[0];
				vec_OrangeSide[7] = tmp_vec[3];
				vec_OrangeSide[8] = tmp_vec[6];
			}

			// Around RedSide
			vec_WhiteSide[6] = vec_OrangeSide[0];
			vec_WhiteSide[7] = vec_OrangeSide[1];
			vec_WhiteSide[8] = vec_OrangeSide[2];

			vec_BlueSide[6] = vec_OrangeSide[2];
			vec_BlueSide[7] = vec_OrangeSide[5];
			vec_BlueSide[8] = vec_OrangeSide[8];

			vec_YellowSide[8] = vec_OrangeSide[8];
			vec_YellowSide[7] = vec_OrangeSide[7];
			vec_YellowSide[6] = vec_OrangeSide[6];

			vec_GreenSide[8] = vec_OrangeSide[6];
			vec_GreenSide[7] = vec_OrangeSide[3];
			vec_GreenSide[6] = vec_OrangeSide[0];
			
			// Switching Axis
			vec_X_Axis[6] = vec_OrangeSide[3];
			vec_X_Axis[8] = vec_OrangeSide[5];

			vec_Y_Axis[6] = vec_OrangeSide[1];
			vec_Y_Axis[8] = vec_OrangeSide[7];
			break;
		}
		case 'g':
		{
			vecBaseVertex tmp_vec = vec_GreenSide;

			if(rotationDirection)
			{
				// Circling GreenSide
				vec_GreenSide[0] = tmp_vec[6];
				vec_GreenSide[1] = tmp_vec[3];
				vec_GreenSide[2] = tmp_vec[0];
				vec_GreenSide[3] = tmp_vec[7];
				vec_GreenSide[4] = tmp_vec[4];
				vec_GreenSide[5] = tmp_vec[1];
				vec_GreenSide[6] = tmp_vec[8];
				vec_GreenSide[7] = tmp_vec[5];
				vec_GreenSide[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling GreenSide
				vec_GreenSide[0] = tmp_vec[2];
				vec_GreenSide[1] = tmp_vec[5];
				vec_GreenSide[2] = tmp_vec[8];
				vec_GreenSide[3] = tmp_vec[1];
				vec_GreenSide[4] = tmp_vec[4];
				vec_GreenSide[5] = tmp_vec[7];
				vec_GreenSide[6] = tmp_vec[0];
				vec_GreenSide[7] = tmp_vec[3];
				vec_GreenSide[8] = tmp_vec[6];
			}

			// Around GreenSide
			vec_WhiteSide[6] = vec_GreenSide[6];
			vec_WhiteSide[3] = vec_GreenSide[3];
			vec_WhiteSide[0] = vec_GreenSide[0];

			vec_RedSide[0] = vec_GreenSide[0];
			vec_RedSide[3] = vec_GreenSide[1];
			vec_RedSide[6] = vec_GreenSide[2];

			vec_YellowSide[0] = vec_GreenSide[2];
			vec_YellowSide[3] = vec_GreenSide[5];
			vec_YellowSide[6] = vec_GreenSide[8];

			vec_OrangeSide[6] = vec_GreenSide[8];
			vec_OrangeSide[3] = vec_GreenSide[7];
			vec_OrangeSide[0] = vec_GreenSide[6];
			
			// Switching Axis
			vec_X_Axis[0] = vec_GreenSide[1];
			vec_X_Axis[6] = vec_GreenSide[7];

			vec_Z_Axis[0] = vec_GreenSide[3];
			vec_Z_Axis[6] = vec_GreenSide[5];
			break;
		}
	case 'u':
		{
			vecBaseVertex tmp_vec = vec_Y_Axis;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_Y_Axis[0] = tmp_vec[6];
				vec_Y_Axis[1] = tmp_vec[3];
				vec_Y_Axis[2] = tmp_vec[0];
				vec_Y_Axis[3] = tmp_vec[7];
				vec_Y_Axis[4] = tmp_vec[4];
				vec_Y_Axis[5] = tmp_vec[1];
				vec_Y_Axis[6] = tmp_vec[8];
				vec_Y_Axis[7] = tmp_vec[5];
				vec_Y_Axis[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_Y_Axis[0] = tmp_vec[2];
				vec_Y_Axis[1] = tmp_vec[5];
				vec_Y_Axis[2] = tmp_vec[8];
				vec_Y_Axis[3] = tmp_vec[1];
				vec_Y_Axis[4] = tmp_vec[4];
				vec_Y_Axis[5] = tmp_vec[7];
				vec_Y_Axis[6] = tmp_vec[0];
				vec_Y_Axis[7] = tmp_vec[3];
				vec_Y_Axis[8] = tmp_vec[6];
			}

			// Around RedSide
			vec_WhiteSide[7] = vec_Y_Axis[6];
			vec_WhiteSide[4] = vec_Y_Axis[3];
			vec_WhiteSide[1] = vec_Y_Axis[0];

			vec_RedSide[1] = vec_Y_Axis[0];
			vec_RedSide[4] = vec_Y_Axis[1];
			vec_RedSide[7] = vec_Y_Axis[2];

			vec_YellowSide[1] = vec_Y_Axis[2];
			vec_YellowSide[4] = vec_Y_Axis[5];
			vec_YellowSide[7] = vec_Y_Axis[8];

			vec_OrangeSide[7] = vec_Y_Axis[8];
			vec_OrangeSide[4] = vec_Y_Axis[7];
			vec_OrangeSide[1] = vec_Y_Axis[6];
			
			// Switching Axis
			vec_X_Axis[1] = vec_Y_Axis[1];
			vec_X_Axis[7] = vec_Y_Axis[7];

			vec_Z_Axis[1] = vec_Y_Axis[3];
			vec_Z_Axis[7] = vec_Y_Axis[5];
			break;
		}
	case 'b':
		{
			vecBaseVertex tmp_vec = vec_BlueSide;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_BlueSide[0] = tmp_vec[6];
				vec_BlueSide[1] = tmp_vec[3];
				vec_BlueSide[2] = tmp_vec[0];
				vec_BlueSide[3] = tmp_vec[7];
				vec_BlueSide[4] = tmp_vec[4];
				vec_BlueSide[5] = tmp_vec[1];
				vec_BlueSide[6] = tmp_vec[8];
				vec_BlueSide[7] = tmp_vec[5];
				vec_BlueSide[8] = tmp_vec[2];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_BlueSide[0] = tmp_vec[2];
				vec_BlueSide[1] = tmp_vec[5];
				vec_BlueSide[2] = tmp_vec[8];
				vec_BlueSide[3] = tmp_vec[1];
				vec_BlueSide[4] = tmp_vec[4];
				vec_BlueSide[5] = tmp_vec[7];
				vec_BlueSide[6] = tmp_vec[0];
				vec_BlueSide[7] = tmp_vec[3];
				vec_BlueSide[8] = tmp_vec[6];
			}

			// Around RedSide
			vec_WhiteSide[8] = vec_BlueSide[6];
			vec_WhiteSide[5] = vec_BlueSide[3];
			vec_WhiteSide[2] = vec_BlueSide[0];

			vec_RedSide[2] = vec_BlueSide[0];
			vec_RedSide[5] = vec_BlueSide[1];
			vec_RedSide[8] = vec_BlueSide[2];

			vec_YellowSide[2] = vec_BlueSide[2];
			vec_YellowSide[5] = vec_BlueSide[5];
			vec_YellowSide[8] = vec_BlueSide[8];

			vec_OrangeSide[8] = vec_BlueSide[8];
			vec_OrangeSide[5] = vec_BlueSide[7];
			vec_OrangeSide[2] = vec_BlueSide[6];
			
			// Switching Axis
			vec_X_Axis[2] = vec_BlueSide[1];
			vec_X_Axis[8] = vec_BlueSide[7];

			vec_Z_Axis[2] = vec_BlueSide[3];
			vec_Z_Axis[8] = vec_BlueSide[5];
			break;
		}
	case 'w':
		{
			vecBaseVertex tmp_vec = vec_WhiteSide;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_WhiteSide[0] = tmp_vec[2];
				vec_WhiteSide[1] = tmp_vec[5];
				vec_WhiteSide[2] = tmp_vec[8];
				vec_WhiteSide[3] = tmp_vec[1];
				vec_WhiteSide[4] = tmp_vec[4];
				vec_WhiteSide[5] = tmp_vec[7];
				vec_WhiteSide[6] = tmp_vec[0];
				vec_WhiteSide[7] = tmp_vec[3];
				vec_WhiteSide[8] = tmp_vec[6];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_WhiteSide[0] = tmp_vec[6];
				vec_WhiteSide[1] = tmp_vec[3];
				vec_WhiteSide[2] = tmp_vec[0];
				vec_WhiteSide[3] = tmp_vec[7];
				vec_WhiteSide[4] = tmp_vec[4];
				vec_WhiteSide[5] = tmp_vec[1];
				vec_WhiteSide[6] = tmp_vec[8];
				vec_WhiteSide[7] = tmp_vec[5];
				vec_WhiteSide[8] = tmp_vec[2];
			}

			// Around RedSide
			vec_OrangeSide[0] = vec_WhiteSide[6];
			vec_OrangeSide[1] = vec_WhiteSide[7];
			vec_OrangeSide[2] = vec_WhiteSide[8];

			vec_BlueSide[6] = vec_WhiteSide[8];
			vec_BlueSide[3] = vec_WhiteSide[5];
			vec_BlueSide[0] = vec_WhiteSide[2];

			vec_RedSide[2] = vec_WhiteSide[2];
			vec_RedSide[1] = vec_WhiteSide[1];
			vec_RedSide[0] = vec_WhiteSide[0];

			vec_GreenSide[0] = vec_WhiteSide[0];
			vec_GreenSide[3] = vec_WhiteSide[3];
			vec_GreenSide[6] = vec_WhiteSide[6];
			
			// Switching Axis
			vec_Y_Axis[0] = vec_WhiteSide[1];
			vec_Y_Axis[6] = vec_WhiteSide[7];

			vec_Z_Axis[0] = vec_WhiteSide[3];
			vec_Z_Axis[2] = vec_WhiteSide[5];
			break;
		}
	case 'x':
		{
			vecBaseVertex tmp_vec = vec_X_Axis;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_X_Axis[0] = tmp_vec[2];
				vec_X_Axis[1] = tmp_vec[5];
				vec_X_Axis[2] = tmp_vec[8];
				vec_X_Axis[3] = tmp_vec[1];
				vec_X_Axis[4] = tmp_vec[4];
				vec_X_Axis[5] = tmp_vec[7];
				vec_X_Axis[6] = tmp_vec[0];
				vec_X_Axis[7] = tmp_vec[3];
				vec_X_Axis[8] = tmp_vec[6];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_X_Axis[0] = tmp_vec[6];
				vec_X_Axis[1] = tmp_vec[3];
				vec_X_Axis[2] = tmp_vec[0];
				vec_X_Axis[3] = tmp_vec[7];
				vec_X_Axis[4] = tmp_vec[4];
				vec_X_Axis[5] = tmp_vec[1];
				vec_X_Axis[6] = tmp_vec[8];
				vec_X_Axis[7] = tmp_vec[5];
				vec_X_Axis[8] = tmp_vec[2];
			}

			// Around RedSide
			vec_OrangeSide[3] = vec_X_Axis[6];
			vec_OrangeSide[4] = vec_X_Axis[7];
			vec_OrangeSide[5] = vec_X_Axis[8];

			vec_BlueSide[7] = vec_X_Axis[8];
			vec_BlueSide[4] = vec_X_Axis[5];
			vec_BlueSide[1] = vec_X_Axis[2];

			vec_RedSide[5] = vec_X_Axis[2];
			vec_RedSide[4] = vec_X_Axis[1];
			vec_RedSide[3] = vec_X_Axis[0];

			vec_GreenSide[1] = vec_X_Axis[0];
			vec_GreenSide[4] = vec_X_Axis[3];
			vec_GreenSide[7] = vec_X_Axis[6];
			
			// Switching Axis
			vec_Y_Axis[1] = vec_X_Axis[1];
			vec_Y_Axis[7] = vec_X_Axis[7];

			vec_Z_Axis[3] = vec_X_Axis[3];
			vec_Z_Axis[5] = vec_X_Axis[5];
			break;
		}
	case 'y':
		{
			vecBaseVertex tmp_vec = vec_YellowSide;

			if(rotationDirection)
			{
				// Circling RedSide
				vec_YellowSide[0] = tmp_vec[2];
				vec_YellowSide[1] = tmp_vec[5];
				vec_YellowSide[2] = tmp_vec[8];
				vec_YellowSide[3] = tmp_vec[1];
				vec_YellowSide[4] = tmp_vec[4];
				vec_YellowSide[5] = tmp_vec[7];
				vec_YellowSide[6] = tmp_vec[0];
				vec_YellowSide[7] = tmp_vec[3];
				vec_YellowSide[8] = tmp_vec[6];
			}
			if(!rotationDirection)
			{
				// Circling RedSide
				vec_YellowSide[0] = tmp_vec[6];
				vec_YellowSide[1] = tmp_vec[3];
				vec_YellowSide[2] = tmp_vec[0];
				vec_YellowSide[3] = tmp_vec[7];
				vec_YellowSide[4] = tmp_vec[4];
				vec_YellowSide[5] = tmp_vec[1];
				vec_YellowSide[6] = tmp_vec[8];
				vec_YellowSide[7] = tmp_vec[5];
				vec_YellowSide[8] = tmp_vec[2];
			}

			// Around RedSide
			vec_OrangeSide[6] = vec_YellowSide[6];
			vec_OrangeSide[7] = vec_YellowSide[7];
			vec_OrangeSide[8] = vec_YellowSide[8];

			vec_BlueSide[8] = vec_YellowSide[8];
			vec_BlueSide[5] = vec_YellowSide[5];
			vec_BlueSide[2] = vec_YellowSide[2];

			vec_RedSide[8] = vec_YellowSide[2];
			vec_RedSide[7] = vec_YellowSide[1];
			vec_RedSide[6] = vec_YellowSide[0];

			vec_GreenSide[2] = vec_YellowSide[0];
			vec_GreenSide[5] = vec_YellowSide[3];
			vec_GreenSide[8] = vec_YellowSide[6];
			
			// Switching Axis
			vec_Y_Axis[2] = vec_YellowSide[1];
			vec_Y_Axis[8] = vec_YellowSide[7];

			vec_Z_Axis[6] = vec_YellowSide[3];
			vec_Z_Axis[8] = vec_YellowSide[5];
			break;
		}
	default:
		{
			break;
		}
	}
}

/* Create libgxm scenes */
void createGxmData( void )
{
/* ---------------------------------------------------------------------
	4. Create a shader patcher and register programs

	A shader patcher object is required to produce vertex and fragment
	programs from the shader compiler output.  First we create a shader
	patcher instance, using callback functions to allow it to allocate
	and free host memory for internal state.

	In order to create vertex and fragment programs for a particular
	shader, the compiler output must first be registered to obtain an ID
	for that shader.  Within a single ID, vertex and fragment programs
	are reference counted and could be shared if created with identical
	parameters.  To maximise this sharing, programs should only be
	registered with the shader patcher once if possible, so we will do
	this now.
	--------------------------------------------------------------------- */

	/* register programs with the patcher */
	int returnCode = sceGxmShaderPatcherRegisterProgram( s_shaderPatcher, &binaryClearVGxpStart, &s_clearVertexProgramId );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );
	returnCode = sceGxmShaderPatcherRegisterProgram( s_shaderPatcher, &binaryClearFGxpStart, &s_clearFragmentProgramId );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );


    returnCode = sceGxmShaderPatcherRegisterProgram( s_shaderPatcher, &binaryBasicVGxpStart, &s_basicVertexProgramId );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );
	returnCode = sceGxmShaderPatcherRegisterProgram( s_shaderPatcher, &binaryBasicFGxpStart, &s_basicFragmentProgramId );
    SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );


/* ---------------------------------------------------------------------
	5. Create the programs and data for the clear

	On SGX hardware, vertex programs must perform the unpack operations
	on vertex data, so we must define our vertex formats in order to
	create the vertex program.  Similarly, fragment programs must be
	specialized based on how they output their pixels and MSAA mode
	(and texture format on ES1).

	We define the clear geometry vertex format here and create the vertex
	and fragment program.

	The clear vertex and index data is static, we allocate and write the
	data here.
	--------------------------------------------------------------------- */

	/* get attributes by name to create vertex format bindings */
	const SceGxmProgram *clearProgram = sceGxmShaderPatcherGetProgramFromId( s_clearVertexProgramId );
	SCE_DBG_ALWAYS_ASSERT( clearProgram );
	const SceGxmProgramParameter *paramClearPositionAttribute = sceGxmProgramFindParameterByName( clearProgram, "aPosition" );
	SCE_DBG_ALWAYS_ASSERT( paramClearPositionAttribute && ( sceGxmProgramParameterGetCategory(paramClearPositionAttribute) == SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE ) );

	/* create clear vertex format */
	SceGxmVertexAttribute clearVertexAttributes[1];
	SceGxmVertexStream clearVertexStreams[1];
	clearVertexAttributes[0].streamIndex = 0;
	clearVertexAttributes[0].offset = 0;
	clearVertexAttributes[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	clearVertexAttributes[0].componentCount = 2;
	clearVertexAttributes[0].regIndex = sceGxmProgramParameterGetResourceIndex( paramClearPositionAttribute );
	clearVertexStreams[0].stride = sizeof(ClearVertex);
	clearVertexStreams[0].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	/* create clear programs */
	returnCode = sceGxmShaderPatcherCreateVertexProgram( s_shaderPatcher, s_clearVertexProgramId, clearVertexAttributes, 1, clearVertexStreams, 1, &s_clearVertexProgram );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	returnCode = sceGxmShaderPatcherCreateFragmentProgram( s_shaderPatcher, s_clearFragmentProgramId,
														   SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4, SCE_GXM_MULTISAMPLE_NONE, NULL,
														   sceGxmShaderPatcherGetProgramFromId(s_clearVertexProgramId), &s_clearFragmentProgram );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* create the clear triangle vertex/index data */
	s_clearVertices = (ClearVertex *)graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, 3 * sizeof(ClearVertex), 4, SCE_GXM_MEMORY_ATTRIB_READ, &s_clearVerticesUId );
	s_clearIndices = (uint16_t *)graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, 3 * sizeof(uint16_t), 2, SCE_GXM_MEMORY_ATTRIB_READ, &s_clearIndicesUId );

	s_clearVertices[0].x = -1.0f;
	s_clearVertices[0].y = -1.0f;
	s_clearVertices[1].x =  3.0f;
	s_clearVertices[1].y = -1.0f;
	s_clearVertices[2].x = -1.0f;
	s_clearVertices[2].y =  3.0f;

	s_clearIndices[0] = 0;
	s_clearIndices[1] = 1;
	s_clearIndices[2] = 2;

    // !! All related to triangle.

    /* get attributes by name to create vertex format bindings */
	/* first retrieve the underlying program to extract binding information */
	const SceGxmProgram *basicProgram = sceGxmShaderPatcherGetProgramFromId( s_basicVertexProgramId );
	SCE_DBG_ALWAYS_ASSERT( basicProgram );
	const SceGxmProgramParameter *paramBasicPositionAttribute = sceGxmProgramFindParameterByName( basicProgram, "aPosition" );
	SCE_DBG_ALWAYS_ASSERT( paramBasicPositionAttribute && ( sceGxmProgramParameterGetCategory(paramBasicPositionAttribute) == SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE ) );
	const SceGxmProgramParameter *paramBasicColorAttribute = sceGxmProgramFindParameterByName( basicProgram, "aColor" );
	SCE_DBG_ALWAYS_ASSERT( paramBasicColorAttribute && ( sceGxmProgramParameterGetCategory(paramBasicColorAttribute) == SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE ) );
	// Normal for Vertices
	const SceGxmProgramParameter *paramBasicNormalAttribute = sceGxmProgramFindParameterByName( basicProgram, "aNormal" );
	SCE_DBG_ALWAYS_ASSERT( paramBasicNormalAttribute && ( sceGxmProgramParameterGetCategory(paramBasicNormalAttribute) == SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE ) );

	const SceGxmProgramParameter *paramBasicUVAttribute = sceGxmProgramFindParameterByName( basicProgram, "aUV" );
	SCE_DBG_ALWAYS_ASSERT( paramBasicUVAttribute && ( sceGxmProgramParameterGetCategory(paramBasicUVAttribute) == SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE ) );

	/* create shaded triangle vertex format */
	SceGxmVertexAttribute basicVertexAttributes[4];
	SceGxmVertexStream basicVertexStreams[1];

	// aPosition
	basicVertexAttributes[0].streamIndex = 0;
	basicVertexAttributes[0].offset = 0; // start at Byte 0
	basicVertexAttributes[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	basicVertexAttributes[0].componentCount = 3; // x y z 
	basicVertexAttributes[0].regIndex = sceGxmProgramParameterGetResourceIndex( paramBasicPositionAttribute );

	// aColor
	basicVertexAttributes[1].streamIndex = 0;
	basicVertexAttributes[1].offset = 12; // start at Byte 12
	basicVertexAttributes[1].format = SCE_GXM_ATTRIBUTE_FORMAT_U8N; // Mapping relation clarified.
	basicVertexAttributes[1].componentCount = 4; // alpha r g b
	basicVertexAttributes[1].regIndex = sceGxmProgramParameterGetResourceIndex( paramBasicColorAttribute );

	// aNormal
	basicVertexAttributes[2].streamIndex = 0;
	basicVertexAttributes[2].offset = 16; // start at Byte 16
	basicVertexAttributes[2].format = SCE_GXM_ATTRIBUTE_FORMAT_F32; // 8 byte
	basicVertexAttributes[2].componentCount = 3; // x y z 
	basicVertexAttributes[2].regIndex = sceGxmProgramParameterGetResourceIndex( paramBasicNormalAttribute );

	// aUV
	basicVertexAttributes[3].streamIndex = 0;
	basicVertexAttributes[3].offset = 28; // start at Byte 16
	basicVertexAttributes[3].format = SCE_GXM_ATTRIBUTE_FORMAT_F32; // 8 byte
	basicVertexAttributes[3].componentCount = 2; // x y z 
	basicVertexAttributes[3].regIndex = sceGxmProgramParameterGetResourceIndex( paramBasicUVAttribute );

	basicVertexStreams[0].stride = sizeof(BasicVertex);
	basicVertexStreams[0].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	/* create shaded triangle shaders */
	returnCode = sceGxmShaderPatcherCreateVertexProgram( s_shaderPatcher, s_basicVertexProgramId, basicVertexAttributes, 4,
														 basicVertexStreams, 1, &s_basicVertexProgram );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	returnCode = sceGxmShaderPatcherCreateFragmentProgram( s_shaderPatcher, s_basicFragmentProgramId,
														   SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4, SCE_GXM_MULTISAMPLE_NONE, NULL,
														   sceGxmShaderPatcherGetProgramFromId(s_basicVertexProgramId), &s_basicFragmentProgram );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* find vertex uniforms by name and cache parameter information */
	s_wvpParam = sceGxmProgramFindParameterByName( basicProgram, "worldViewProjection" );
	SCE_DBG_ALWAYS_ASSERT( s_wvpParam && ( sceGxmProgramParameterGetCategory( s_wvpParam ) == SCE_GXM_PARAMETER_CATEGORY_UNIFORM ) );

	/* create shaded triangle vertex/index data */
	s_basicVertices = (BasicVertex *)graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, 4 * 6 * 9 * 3 * sizeof(BasicVertex), 4, SCE_GXM_MEMORY_ATTRIB_READ, &s_basicVerticesUId );
	s_basicIndices = (uint16_t *)graphicsAlloc( SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, 6 * 6 * 9 * 3 * sizeof(uint16_t), 2, SCE_GXM_MEMORY_ATTRIB_READ, &s_basicIndiceUId );

	int cubeGroupYIndex = 0;
	int cubeGroupZIndex = 0;

	int indicesBaseCount = 0;
	int indicesCount = 0;
	int verticesCount = 0;

	float offsetRow = 0;
	float offsetColumn = 0;
	float offset_Z = 0;

	float offset = 0.40f; // 40px = cubeLength

	// Create 27 Cubes
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			for(int k = 0; k < 3; ++k)
			{
				CreateOneCube(currentCubeIndex, cubeGroupYIndex, cubeGroupZIndex, verticesCount, indicesCount, indicesBaseCount, offsetRow, offsetColumn, offset_Z);
				
				currentCubeIndex++;

				indicesBaseCount += 4 * 6;
				indicesCount += 6 * 6;
				verticesCount += 4 * 6;

				offsetRow += offset;
			}
			cubeGroupYIndex++;

			offsetColumn += offset;
			offsetRow = 0;
		}
		cubeGroupZIndex++;

		offset_Z += offset;
		offsetColumn = 0;
	}

	for(int i = 0; i < 6; ++i)
	{
		vec_normals.push_back(Vector3(	vec_RedSide[0][i*4]->normal[0],
										vec_RedSide[0][i*4]->normal[1],
										vec_RedSide[0][i*4]->normal[2]));
	}
}

/* Main render function */
void render( void )
{
	/* render libgxm scenes */
	renderGxm();
}

/* render gxm scenes */
void renderGxm( void )
{
/* -----------------------------------------------------------------
	8. Rendering step

	This sample renders a single scene containing the clear triangle.
	Before any drawing can take place, a scene must be started.
	We render to the back buffer, so it is also important to use a
	sync object to ensure that these rendering operations are
	synchronized with display operations.

	The clear triangle shaders do not declare any uniform variables,
	so this may be rendered immediately after setting the vertex and
	fragment program.

	Once clear triangle have been drawn the scene can be ended, which
	submits it for rendering on the GPU.
	----------------------------------------------------------------- */

	/* start rendering to the render target */
	sceGxmBeginScene( s_context, 0, s_renderTarget, NULL, NULL, s_displayBufferSync[s_displayBackBufferIndex], &s_displaySurface[s_displayBackBufferIndex], &s_depthSurface );

	/* set clear shaders */
	sceGxmSetVertexProgram( s_context, s_clearVertexProgram );
	sceGxmSetFragmentProgram( s_context, s_clearFragmentProgram );

	/* draw ther clear triangle */
	sceGxmSetVertexStream( s_context, 0, s_clearVertices );
	sceGxmDraw( s_context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, s_clearIndices, 3 );

    // !! Speciality for rendering a triangle.
    /* render the  triangle */
	sceGxmSetVertexProgram( s_context, s_basicVertexProgram );
	sceGxmSetFragmentProgram( s_context, s_basicFragmentProgram );

	/* set the vertex program constants */
	void *vertexDefaultBuffer;
	sceGxmReserveVertexDefaultUniformBuffer( s_context, &vertexDefaultBuffer );
	sceGxmSetUniformDataF( vertexDefaultBuffer, s_wvpParam, 0, 16, (float*)&s_finalTransformation );

	/* draw the spinning triangle */
	sceGxmSetVertexStream( s_context, 0, s_basicVertices );
	sceGxmDraw( s_context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, s_basicIndices, 6 * 6 * 9 * 3); // Indices * faces * cubes * rows

	/* stop rendering to the render target */
	sceGxmEndScene( s_context, NULL, NULL );
}

/* queue a display swap and cycle our buffers */
void cycleDisplayBuffers( void )
{
/* -----------------------------------------------------------------
	9-a. Flip operation

	Now we have finished submitting rendering work for this frame it
	is time to submit a flip operation.  As part of specifying this
	flip operation we must provide the sync objects for both the old
	buffer and the new buffer.  This is to allow synchronization both
	ways: to not flip until rendering is complete, but also to ensure
	that future rendering to these buffers does not start until the
	flip operation is complete.

	Once we have queued our flip, we manually cycle through our back
	buffers before starting the next frame.
	----------------------------------------------------------------- */

	/* PA heartbeat to notify end of frame */
	sceGxmPadHeartbeat( &s_displaySurface[s_displayBackBufferIndex], s_displayBufferSync[s_displayBackBufferIndex] );

	/* queue the display swap for this frame */
	DisplayData displayData;
	displayData.address = s_displayBufferData[s_displayBackBufferIndex];

	/* front buffer is OLD buffer, back buffer is NEW buffer */
	sceGxmDisplayQueueAddEntry( s_displayBufferSync[s_displayFrontBufferIndex], s_displayBufferSync[s_displayBackBufferIndex], &displayData );

	/* update buffer indices */
	s_displayFrontBufferIndex = s_displayBackBufferIndex;
	s_displayBackBufferIndex = (s_displayBackBufferIndex + 1) % DISPLAY_BUFFER_COUNT;
}

/* Destroy Gxm Data */
void destroyGxmData( void )
{
/* ---------------------------------------------------------------------
	11. Destroy the programs and data for the clear and spinning triangle

	Once the GPU is finished, we release all our programs.
	--------------------------------------------------------------------- */

	/* clean up allocations */
	sceGxmShaderPatcherReleaseFragmentProgram( s_shaderPatcher, s_clearFragmentProgram );
	sceGxmShaderPatcherReleaseVertexProgram( s_shaderPatcher, s_clearVertexProgram );
	graphicsFree( s_clearIndicesUId );
	graphicsFree( s_clearVerticesUId );

	/* wait until display queue is finished before deallocating display buffers */
	sceGxmDisplayQueueFinish();

	/* unregister programs and destroy shader patcher */
	sceGxmShaderPatcherUnregisterProgram( s_shaderPatcher, s_clearFragmentProgramId );
	sceGxmShaderPatcherUnregisterProgram( s_shaderPatcher, s_clearVertexProgramId );
	sceGxmShaderPatcherDestroy( s_shaderPatcher );
	fragmentUsseFree( s_patcherFragmentUsseUId );
	vertexUsseFree( s_patcherVertexUsseUId );
	graphicsFree( s_patcherBufferUId );
}

/* ShutDown libgxm */
int shutdownGxm( void )
{
/* ---------------------------------------------------------------------
	12. Finalize libgxm

	Once the GPU is finished, we deallocate all our memory,
	destroy all object and finally terminate libgxm.
	--------------------------------------------------------------------- */

	int returnCode = SCE_OK;

	graphicsFree( s_depthBufferUId );

	for ( unsigned int i = 0 ; i < DISPLAY_BUFFER_COUNT; ++i )
	{
		memset( s_displayBufferData[i], 0, DISPLAY_HEIGHT*DISPLAY_STRIDE_IN_PIXELS*4 );
		graphicsFree( s_displayBufferUId[i] );
		sceGxmSyncObjectDestroy( s_displayBufferSync[i] );
	}

	/* destroy the render target */
	sceGxmDestroyRenderTarget( s_renderTarget );

	/* destroy the context */
	sceGxmDestroyContext( s_context );

	fragmentUsseFree( s_fragmentUsseRingBufferUId );
	graphicsFree( s_fragmentRingBufferUId );
	graphicsFree( s_vertexRingBufferUId );
	graphicsFree( s_vdmRingBufferUId );
	free( s_contextParams.hostMem );

	/* terminate libgxm */
	sceGxmTerminate();
	return returnCode;
}

/* Host alloc */
static void *patcherHostAlloc( void *userData, unsigned int size )
{
	(void)( userData );
	return malloc( size );
}

/* Host free */
static void patcherHostFree( void *userData, void *mem )
{
	(void)( userData );
	free( mem );
}

/* Display callback */
void displayCallback( const void *callbackData )
{
/* -----------------------------------------------------------------
	10-b. Flip operation

	The callback function will be called from an internal thread once
	queued GPU operations involving the sync objects is complete.
	Assuming we have not reached our maximum number of queued frames,
	this function returns immediately.
	----------------------------------------------------------------- */

	SceDisplayFrameBuf framebuf;

	/* cast the parameters back */
	const DisplayData *displayData = (const DisplayData *)callbackData;


    // Render debug text.
    /* set framebuffer info */
	SceDbgFontFrameBufInfo info;
	memset( &info, 0, sizeof(SceDbgFontFrameBufInfo) );
	info.frameBufAddr = (SceUChar8 *)displayData->address;
	info.frameBufPitch = DISPLAY_STRIDE_IN_PIXELS;
	info.frameBufWidth = DISPLAY_WIDTH;
	info.frameBufHeight = DISPLAY_HEIGHT;
	info.frameBufPixelformat = DBGFONT_PIXEL_FORMAT;

	/* flush font buffer */
	int returnCode = sceDbgFontFlush( &info );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );
	

	/* wwap to the new buffer on the next VSYNC */
	memset(&framebuf, 0x00, sizeof(SceDisplayFrameBuf));
	framebuf.size        = sizeof(SceDisplayFrameBuf);
	framebuf.base        = displayData->address;
	framebuf.pitch       = DISPLAY_STRIDE_IN_PIXELS;
	framebuf.pixelformat = DISPLAY_PIXEL_FORMAT;
	framebuf.width       = DISPLAY_WIDTH;
	framebuf.height      = DISPLAY_HEIGHT;
	returnCode = sceDisplaySetFrameBuf( &framebuf, SCE_DISPLAY_UPDATETIMING_NEXTVSYNC );
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );

	/* block this callback until the swap has occurred and the old buffer is no longer displayed */
	returnCode = sceDisplayWaitVblankStart();
	SCE_DBG_ALWAYS_ASSERT( returnCode == SCE_OK );
}

/* Alloc used by libgxm */
static void *graphicsAlloc( SceKernelMemBlockType type, uint32_t size, uint32_t alignment, uint32_t attribs, SceUID *uid )
{
/*	Since we are using sceKernelAllocMemBlock directly, we cannot directly
	use the alignment parameter.  Instead, we must allocate the size to the
	minimum for this memblock type, and just SCE_DBG_ALWAYS_ASSERT that this will cover
	our desired alignment.

	Developers using their own heaps should be able to use the alignment
	parameter directly for more minimal padding.
*/

	if( type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RWDATA )
	{
		/* CDRAM memblocks must be 256KiB aligned */
		SCE_DBG_ALWAYS_ASSERT( alignment <= 256*1024 );
		size = ALIGN( size, 256*1024 );
	}
	else
	{
		/* LPDDR memblocks must be 4KiB aligned */
		SCE_DBG_ALWAYS_ASSERT( alignment <= 4*1024 );
		size = ALIGN( size, 4*1024 );
	}

	/* allocate some memory */
	*uid = sceKernelAllocMemBlock( "simple", type, size, NULL );
	SCE_DBG_ALWAYS_ASSERT( *uid >= SCE_OK );

	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase( *uid, &mem );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* map for the GPU */
	err = sceGxmMapMemory( mem, size, attribs );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* done */
	return mem;
}

/* Free used by libgxm */
static void graphicsFree( SceUID uid )
{
	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase(uid, &mem);
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);

	// unmap memory
	err = sceGxmUnmapMemory(mem);
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);

	// free the memory block
	err = sceKernelFreeMemBlock(uid);
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);
}

/* vertex alloc used by libgxm */
static void *vertexUsseAlloc( uint32_t size, SceUID *uid, uint32_t *usseOffset )
{
	/* align to memblock alignment for LPDDR */
	size = ALIGN( size, 4096 );

	/* allocate some memory */
	*uid = sceKernelAllocMemBlock( "simple", SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, size, NULL );
	SCE_DBG_ALWAYS_ASSERT( *uid >= SCE_OK );

	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase( *uid, &mem );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* map as vertex USSE code for the GPU */
	err = sceGxmMapVertexUsseMemory( mem, size, usseOffset );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	return mem;
}

/* vertex free used by libgxm */
static void vertexUsseFree( SceUID uid )
{
	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase( uid, &mem );
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);

	/* unmap memory */
	err = sceGxmUnmapVertexUsseMemory( mem );
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);

	/* free the memory block */
	err = sceKernelFreeMemBlock( uid );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );
}

/* fragment alloc used by libgxm */
static void *fragmentUsseAlloc( uint32_t size, SceUID *uid, uint32_t *usseOffset )
{
	/* align to memblock alignment for LPDDR */
	size = ALIGN( size, 4096 );

	/* allocate some memory */
	*uid = sceKernelAllocMemBlock( "simple", SCE_KERNEL_MEMBLOCK_TYPE_USER_RWDATA_UNCACHE, size, NULL );
	SCE_DBG_ALWAYS_ASSERT( *uid >= SCE_OK );

	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase( *uid, &mem );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* map as fragment USSE code for the GPU */
	err = sceGxmMapFragmentUsseMemory( mem, size, usseOffset);
	SCE_DBG_ALWAYS_ASSERT(err == SCE_OK);

	// done
	return mem;
}

/* fragment free used by libgxm */
static void fragmentUsseFree( SceUID uid )
{
	/* grab the base address */
	void *mem = NULL;
	int err = sceKernelGetMemBlockBase( uid, &mem );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* unmap memory */
	err = sceGxmUnmapFragmentUsseMemory( mem );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );

	/* free the memory block */
	err = sceKernelFreeMemBlock( uid );
	SCE_DBG_ALWAYS_ASSERT( err == SCE_OK );
}

