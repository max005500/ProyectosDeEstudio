
from dataclasses import dataclass

import numpy as np
import torch
import skimage as sk
from dataclasses import dataclass, field, InitVar
from numpy.typing import NDArray
from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class Kmeanspp:
    img: NDArray
    k: InitVar[int]  # type: ignore
    torch_device: str = "cpu"
    notes = False
    _k: int = 0

    def __post_init__(self,k):
        self.device = torch.device(self.torch_device)

        # Tamaño original M x N
        self.size = self.img[:, :, 0].shape
        self.img = sk.img_as_float32(self.img)

        # Convertir imagen a tensor
        data = torch.from_numpy(self.img).to(self.device)

        # Convertir uint8 [0, 255] a float32 [0, 1]
        data = data.to(torch.float32) 

        # Convertir imagen M x N x 3 a matriz de puntos [M*N, 3]
        self.datos = data.reshape(-1, 3)

        # Filas RGB únicas
        self.unicos, self.ic, self.frecuencias = torch.unique(
            self.datos,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        
        self.frecuencias = self.frecuencias.to(torch.float32)
        self.k = k # type: ignore

        
        # Equivalente a:
        # iteracion = K*10;
        # if K < 10
        #     iteracion = 100;
        # end

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self,k):
        if k > self.unicos.shape[0]:
            raise ValueError(
                f"k={k} es mayor que la cantidad de colores únicos " 
                f"({self.unicos.shape[0]})"
            )

        self.iteracion = max(k * 10, 100)
        self._k = k

        # Inicialización K-means++
        self._kmeansppInit()


    def _kmeansppInit(self):
            # Primer centroide aleatorio
            idx0 = torch.randint(
                0,
                self.unicos.shape[0],
                (1,),
                device=self.device,
            )

            self.C = self.unicos[idx0, :]  # [1, 3]


            for _ in range(1, self._k):
                dist_matrix = torch.cdist(self.unicos, self.C, p=2).pow(2)

                distancias = dist_matrix.min(dim=1).values

                pesos = distancias * self.frecuencias
                suma_pesos = pesos.sum()

                if suma_pesos <= 0:
                    break

                probs = pesos / suma_pesos

                idx = torch.multinomial(probs, num_samples=1)

                new_centroid = self.unicos[idx, :]
                self.C = torch.cat([self.C, new_centroid], dim=0)

    def colorMap(self):
        tol = 0.01

        for iter_idx in range(self.iteracion):
            # Distancia de cada color único a cada centroide
            distancias = torch.cdist(self.unicos, self.C, p=2)

            # Cluster más cercano para cada color único
            asignaciones_unicos = torch.argmin(distancias, dim=1)

            nuevos_centroides = self.C.clone()

            for x in range(self._k):
                mask = asignaciones_unicos == x

                if torch.any(mask):
                    puntos_cluster = self.unicos[mask]
                    pesos_cluster = self.frecuencias[mask].unsqueeze(1)

                    nuevos_centroides[x, :] = (
                        puntos_cluster * pesos_cluster
                    ).sum(dim=0) / pesos_cluster.sum()
                else:
                    nuevos_centroides[x, :] = self.C[x, :]

            # Criterio de convergencia
            if torch.max(torch.abs(self.C - nuevos_centroides)) < tol:
                self.C = nuevos_centroides
                if self.notes:
                    print(f"Convergencia alcanzada en la iteración: {iter_idx + 1}")
                break

            self.C = nuevos_centroides

        else:
            print(
                f"Se necesitan más de {self.iteracion} iteraciones "
                "para llegar a convergencia"
            )

        # Recalcular asignaciones finales
        distancias = torch.cdist(self.unicos, self.C, p=2)
        asignaciones_unicos = torch.argmin(distancias, dim=1)

        # Expandir desde colores únicos a todos los píxeles originales
        asignaciones = asignaciones_unicos[self.ic]

        # Volver a forma M x N
        asignaciones = asignaciones.reshape(self.size).to(torch.uint8)

        return asignaciones, self.C

    def compressedImage(self):
        asignaciones, C = self.colorMap()

        # Reconstruir imagen comprimida M x N x 3
        img_comp = C[asignaciones.long()]

        # Volver de [0, 1] a uint8 [0, 255]
        img_comp = torch.clamp(img_comp * 255.0, 0, 255).to(torch.uint8)
        torch.cuda.empty_cache()

        return img_comp.detach().cpu().numpy(), asignaciones.detach().cpu().numpy(), C.detach().cpu().numpy()