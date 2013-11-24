A = imread('Test1.ppm');
roi = imcrop(A)

roi_hsv = rgb2hsv(roi);
imshow(roi_hsv)
[x y z] = size(roi_hsv)
pixels_hsv = zeros(20, 3);

h = x;
w = y;

for n = 1:1
    i = ceil(rand(1) * h)
    j = ceil(rand(1) * w)
    pixels_hsv(n, :) = roi_hsv(i, j, :);
    roi_hsv(i, j, 1) = [];
    roi_hsv(i, j, 2) = [];
    roi_hsv(i, j, 3) = [];
    
end
[x y z] = size(roi_hsv)