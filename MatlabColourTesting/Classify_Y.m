clear all;
close all;

test = imread('Test1.ppm');
figure, imshow(test)
x = 480;
y = 640;
test = rgb2ycbcr(test);
result = test;
numPixels = x * y;
ycbcr = zeros(numPixels, 3);

DG_Cb = 128.599; %Dark Green Means
DG_Cr = 112.663; 
DG_Y = 82.581;

LG_Cb = 128.584; %Light Green Means
LG_Cr = 109.507;
LG_Y = 133.502;

R_Cb = 130.819; %Red Means
R_Cr = 145.577;
R_Y = 74.066;


for i = 1:x
    for j = 1:y
        Y = double(test(i, j, 1));
        Cb = double(test(i, j, 2));
        Cr = double(test(i, j, 3));
    
        Y_dist = abs(Y - R_Y);
        R_dist = sqrt(((Cb - R_Cb)^2) + ((Cr - R_Cr)^2));
        
        if R_dist < 10 && Y_dist < 25
            result(i, j, 1) = R_Y;
            result(i, j, 2) = R_Cb;
            result(i, j, 3) = R_Cr;
        else
            result(i, j, 1) = 16;
            result(i, j, 2) = 128;
            result(i, j, 3) = 128;
        end

    end
end
result = ycbcr2rgb(result);
result = im2bw(rgb2gray(result), 0.2);
filled = imfill(result, 'holes');
holes = filled & ~result;
holes = bwareaopen(holes, 10);

filled2 = filled | holes;
figure, imshow(result)
%figure, imshow(filled)