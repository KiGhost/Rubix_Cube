# Rubix_Cube
Rubix Cube developed for the Playstation Vita with fully enabled front and backtouch control

The main.cpp in Rubix_Cube/HelloClear/ is the heart of this application.
It is splitted in these steps:

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

You will find the Vertex- and Fragmentshader in Rubix_Cube/HelloClear/graph/.
