%Means
clear all;
close all;

result = zeros(2, 7);


for Test = 1:7
test = imread('Test1.ppm');
figure, imshow(test)
h = imrect(gca, [10 10 20 20]);
accepted_pos = wait(h);
x = accepted_pos(1);
y = accepted_pos(2);
roi = imcrop(test, [x y 20 20 ]);
figure, imshow(roi)
ycbcr = zeros(900, 3);
roi_ycbcr = rgb2ycbcr(roi);


for i = 1:20
    for j = 1:20
        n = (i-1)*20 + j;
        ycbcr(n, :) = roi_ycbcr(i, j, :);
    end
end

for z = 1:400
 Cb = ycbcr(z, 2);
 Cr = ycbcr(z, 3);
 
 %Test = 7;  % 1=DG, 2=R, 3=Y, 4=P, 5=Pi, 6=LG, 7=B
 if Test == 1
     result(1, 1) = result(1, 1) + Cb;
     result(2, 1) = result(2, 1) + Cr;
 elseif Test == 2
     result(1, 2) = result(1, 2) + Cb;
     result(2, 2) = result(2, 2) + Cr; 
 elseif Test == 3
     result(1, 3) = result(1, 3) + Cb;
     result(2, 3) = result(2, 3) + Cr; 
 elseif Test == 4
     result(1, 4) = result(1, 4) + Cb;
     result(2, 4) = result(2, 4) + Cr; 
 elseif Test == 5
     result(1, 5) = result(1, 5) + Cb;
     result(2, 5) = result(2, 5) + Cr; 
 elseif Test == 6
     result(1, 6) = result(1, 6) + Cb;
     result(2, 6) = result(2, 6) + Cr; 
 elseif Test == 7
     result(1, 7) = result(1, 7) + Cb;
     result(2, 7) = result(2, 7) + Cr; 
 end
 end
    
end

result = result./400