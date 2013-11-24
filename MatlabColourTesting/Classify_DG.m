
clear all;
close all;

test = imread('Test4_2.ppm');
figure, imshow(test)
imwrite(test, 'Test4_2.jpg', 'JPG');
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

for i = 1:x
    for j = 1:y
        Y = double(test(i, j, 1));
        Cb = double(test(i, j, 2));
        Cr = double(test(i, j, 3));
    
        Y_dist = abs(Y - LG_Y);
        LG_dist = sqrt(((Cb - LG_Cb)^2) + ((Cr - LG_Cr)^2));
        
        if LG_dist < 8 && Y_dist < 100
            result(i, j, 1) = LG_Y;
            result(i, j, 2) = LG_Cb;
            result(i, j, 3) = LG_Cr;
        else
            result(i, j, 1) = 16;
            result(i, j, 2) = 128;
            result(i, j, 3) = 128;
        end

    end
end
result = ycbcr2rgb(result);
result = ycbcr2rgb(result);
result = im2bw(rgb2gray(result), 0.4);
figure, imshow(result)