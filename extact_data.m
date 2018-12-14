% Run this file after opening the FIG file 
h = gcf;
axesObjs = get(h, 'Children');  %axes handles
dataObjs = get(axesObjs, 'Children'); %handles to low-level graphics objects in axes
xdata = get(dataObjs, 'XData');
ydata = get(dataObjs, 'YData');