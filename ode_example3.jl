"""
ODE equation exact solution and numerical solution using different methods
dy1/dt = y2
dy2/dt = -ω^2*y1
Exact Solution:
y1(t) = cos(ωt); y2(t) = sin(ωt)
Numerical solution - Euler forward scheme
y1(t+dt) = y1(t) + dt*y2
y2(t+dt) = y2(t) + dt*(-ω^2*y1)
"""
# using PyPlot
using Printf

dt = 0.005
t = 0.0
T = 1.0
nt = Int64(T/dt)
omega = 2.0*pi
y1exact = zeros(Float64, nt+1)
y2exact = zeros(Float64, nt+1)
y1euler = zeros(Float64, nt+1)
y2euler = zeros(Float64, nt+1)
tseries = zeros(Float64, nt+1)
for i = 1:nt+1
    global t
    tseries[i] = t
    y1exact[i] = cos(omega*t)
    y2exact[i] = -omega*sin(omega*t)
    if i == 1
        y1euler[i] = 1.0
        y2euler[i] = 0.0
    else
        # y1euler[i] = y1euler[i-1] + dt*(-omega*sin(omega*(t-dt)))
        # y2euler[i] = y2euler[i-1] + dt*(-1.0*omega*omega*cos(omega*(t-dt)))
        y1euler[i] = y1euler[i-1] + dt*y2euler[i-1]
        y2euler[i] = y2euler[i-1] + dt*(-1.0*omega*omega*y1euler[i-1])
    end
    t = t+dt
end
plt1 = plot(tseries, y1exact, linewidth=2, reuse = false, label = "Exact y1")
display(plt1)
plt2 = plot(tseries, y1euler, linewidth=2, reuse = false, label = "Euler y1")
display(plt2)
plt3 = plot(tseries, y2exact, linewidth=2, reuse = false, label = "Exact y2")
display(plt3)
plt4 = plot(tseries, y2euler, linewidth=2, reuse = false, label = "Euler y2")
display(plt4)

output = open("output.txt", "w");
for i = 1:nt+1
    write(output, @sprintf("%.16f",y1exact[i])," ", @sprintf("%.16f", y2exact[i]), " ",
          @sprintf("%.16f", y1euler[i])," ", @sprintf("%.16f", y2euler[i])," ", "\n")
end
close(output);
