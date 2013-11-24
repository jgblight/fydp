%NN Classifier Test
clear all;
close all;

result = zeros(7, 7);
Means = [129.67	131.8675	114.0275	150.19	136.765	129.9425	158.055; 
    111.7725	143.525	128.1125	124.4125	146.045	108.4575	93.895];

DG_Cb = Means(1, 1); %Dark Green Means
DG_Cr = Means(2, 1); 
R_Cb = Means(1, 2); %Red Means
R_Cr = Means(2, 2);
Y_Cb = Means(1, 3); %Yellow Means
Y_Cr = Means(2, 3);
P_Cb = Means(1, 4); %Purple Means
P_Cr = Means(2, 4);
Pi_Cb = Means(1, 5); %Pink Means
Pi_Cr = Means(2, 5);
LG_Cb = Means(1, 6); %Light Green Means
LG_Cr = Means(2, 6);
B_Cb = Means(1, 7); % Blue Means
B_Cr = Means(2, 7);

for Test = 1:7
test = imread('Test1_2.ppm');
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
 
 DG_dist = sqrt(((Cb - DG_Cb)^2) + ((Cr - DG_Cr)^2));
 R_dist = sqrt(((Cb - R_Cb)^2) + ((Cr - R_Cr)^2));
 Y_dist = sqrt(((Cb - Y_Cb)^2) + ((Cr - Y_Cr)^2));
 P_dist = sqrt(((Cb - P_Cb)^2) + ((Cr - P_Cr)^2));
 Pi_dist = sqrt(((Cb - Pi_Cb)^2) + ((Cr - Pi_Cr)^2));
 LG_dist = sqrt(((Cb - LG_Cb)^2) + ((Cr - LG_Cr)^2));
 B_dist = sqrt(((Cb - B_Cb)^2) + ((Cr - B_Cr)^2));
 
 dists = [DG_dist, R_dist, Y_dist, P_dist, Pi_dist, LG_dist, B_dist]

 Class = min(dists);
 
 %Test = 7;  % 1=DG, 2=R, 3=Y, 4=P, 5=Pi, 6=LG, 7=B
 if Class == DG_dist
     result(Test, 1) = result(Test, 1) + 1;
 elseif Class == R_dist
     result(Test, 2) = result(Test, 2) + 1;
 elseif Class == Y_dist
     result(Test, 3) = result(Test, 3) + 1;
 elseif Class == P_dist
     result(Test, 4) = result(Test, 4) + 1;
 elseif Class == Pi_dist
     result(Test, 5) = result(Test, 5) + 1;
 elseif Class == LG_dist
     result(Test, 6) = result(Test, 6) + 1;
 elseif Class == B_dist
     result(Test, 7) = result(Test, 7) + 1;
 end
    
end
end

result