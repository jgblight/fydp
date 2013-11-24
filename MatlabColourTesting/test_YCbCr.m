clear all;
close all;

A = imread('Test1.ppm');
figure, imshow(A)
h = imrect(gca, [10 10 30 30]);
accepted_pos = wait(h)
x = accepted_pos(1);
y = accepted_pos(2);
roi = imcrop(A, [x y 30 30 ]);
figure, imshow(roi)

roi_ycbcr = rgb2ycbcr(roi);

ycbcr = zeros(900, 3);

for i = 1:30
    for j = 1:30
        n = (i-1)*30 + j;
        ycbcr(n, :) = roi_ycbcr(i, j, :);
    end
end

