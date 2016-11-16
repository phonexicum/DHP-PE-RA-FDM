#! /usr/bin/gnuplot

# ============================================================================= FILES
data='output.dat'
fout='picture.png'

# ============================================================================= TERMINAL
#set terminal wxt size 1600,900 font ',14'
set terminal wxt size 1000,600 font ',14'
#set term png size 1600,900 font ',14'
#set output fout

# ============================================================================= VIEW
# View for function value
set view 60,13

# View for absolute error
#set view 53, 287
#set view 50, 225

# View for relative error
#set view 50, 280


# ============================================================================= TITLE
set title '' font ",26"
set xlabel "x"
set ylabel "y"
set zlabel "u(x, y)" offset graph 0,0,0.7

# ============================================================================= STYLES
set style line 1 lt 1 lw 1 lc rgb '#ff0000' pt -1

set ticslevel 0
set grid back lt 0 linewidth 0.500, lt 0 linewidth 0.500

set pm3d implicit at s
set pm3d scansforward
#set pm3d interpolate 20,20 flush begin ftriangles noborder corners2color mean
#set pm3d interpolate 2,2 flush begin ftriangles noborder corners2color mean

set pal maxcolors 0
set palette defined

# ============================================================================= PLOT IT !
#plot data title '' ls 1
splot data title '' ls 1
#splot data title '' with lines ls 1

pause mouse button3
