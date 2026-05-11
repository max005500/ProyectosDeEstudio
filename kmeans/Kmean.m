%% Maximiliano Agustin Torres Marin
clc;clear;close all %<== limpieza
%% Inicio de codigo.

% NOTE: Ingreso de datos de entrada
imagen = imread("imagenes/img3.jpg");

K = uint8(180);

tic
[imagenComp,image_IDX] = COLORCOMP(imagen,K);
newImg = COLORDECOMP(imagenComp,image_IDX);
toc

%% Ejecucion y pruebas.

% Calcula el SSIM
[ssimval, ssimmap] = ssim(newImg, imagen);
MSE = immse(imagen, newImg);
PSNR = psnr(imagen, newImg);

figure
imshowpair(ssimmap,imagen,'montage');
figure
imshowpair(newImg,imagen,'montage'); % Muestra la imagen comprimida en RGB

metrica = ["PSNR";"MSE";"SSIM"];
valor = [PSNR;MSE;ssimval];
table(metrica,valor)

%% Funciones.

%NOTE: Compresion de colores
function [asignaciones, C] = COLORCOMP(imagen,K)
%NOTE: Conversion de matriz de MxNx3 a un vector de puntos en 3 dimensiones.

tam = size(imagen(:,:,1));
datos = reshape(im2double(imagen), [], 3);
[unicos, ~, ic] = unique(datos, 'rows');
frecuencias = accumarray(ic, 1);

%NOTE: busqueda de primeros centroides mediante aplicacion de kmeans++

C = unicos(randi(size(unicos,1)), :);
for i = 2:K
    distancias = min(pdist2(unicos, C, "squaredeuclidean"),[],2);
    probs = (distancias.* frecuencias) / sum(distancias.* frecuencias);
    cumprobs = cumsum(probs);
    r = rand();
    new_centroid = unicos(find(cumprobs >= r, 1), :);
    C = [C; new_centroid];
end
%NOTE: Algoritmo K-means con 100 iteraciones(Maximo).
iteracion = K*10;
if K < 10
    iteracion = 100;
end

for iter = 1:iteracion
    %NOTE: Asignación de puntos a los centroides más cercanos o
    %clusters
    
    distancias = pdist2(unicos, C,"euclidean");
    [~, asignaciones_unicos] = min(distancias, [], 2);
    asignaciones = asignaciones_unicos(ic);
    
    for x = 1:K
        cluster = datos(asignaciones == x, :);
        nuevosCentroides(x,:) = mean(cluster, 1);
    end
    
    %NOTE: termino de bucle en caso de convergencia.
    if all(all(C-nuevosCentroides < 0.001))
        disp(['Convergencia alcanzada en la iteración: ', num2str(iter)]);
        break;
    end
    if iter == iteracion-1
        warning("Se necesitan más de "+ iter +" iteraciones para llegar a una convergencia")
    end
    C = nuevosCentroides;
    
end

%NOTE: Matriz nxm de salida
asignaciones = reshape(uint8(asignaciones-1),tam);
end

function [imagen] = COLORDECOMP(img_comp,img_IDX)
tam = [size(img_comp),3]; %<-- tamaño de la imagen
flat_img = reshape(img_comp,[],1);  %<-- imagen aplanada
reconstruct = img_IDX(double(flat_img+1), :); %<-- reconstruccion imagen
imagen =  im2uint8(reshape(reconstruct, tam)); %<-- imagen reconstruida
end