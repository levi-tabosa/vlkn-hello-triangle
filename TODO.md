Clear text field on given action
 - make text field to update on backspace
 - improve  
Move camera with buttons/scroll while on FPS mode

Implement faster UI than current "immediate"-UI

Implement Mesh abstraction

Custom assets (build systam that supports custom textures, models, fonts, etc.. (or even shaders for backgrounds or effects))

Get fonts through fetch system (Google)

Animated transforms
 -
./zig1 lib build-exe -ofmt=c -lc -OReleaseSmall --name zig2 -femit-bin=zig2.c -target x86_64-linux --dep build_options --dep aro -Mroot=src/main.zig -Mbuild_options=config.zig -Maro=lib/compiler/aro/aro.zig
./zig1 lib build-obj -ofmt=c -OReleaseSmall --name compiler_rt -femit-bin=compiler_rt.c -target x86_64-linux -Mroot=lib/compiler_rt.zig
