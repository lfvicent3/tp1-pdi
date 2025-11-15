import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    return (np.array(pA) + np.array(pB)) / 2

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    return indices_delaunay(pontos_medios(pA, pB))

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    return a*t + b

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """
    return 1.0 / (1.0 + np.exp(-2*k * (t - 0.5)))

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    return 0.5

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    det = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    return det

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    v1, v2, v3 = tri

    w = _det3(v1, v2, v3)
    w1 = _det3(pt, v2, v3) / w
    w2 = _det3(v1, pt, v3) / w
    w3 = 1.0 - w1 - w2

    return (w1, w2, w3)


def _check_bari(w1, w2, w3, eps=1e-6):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """
    return (w1 >= -eps) and (w2 >= -eps) and (w3 >= -eps)

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """
    x_min = np.floor(np.min(tri[:, 0])).astype(int)
    x_max = np.ceil(np.max(tri[:, 0])).astype(int)
    y_min = np.floor(np.min(tri[:, 1])).astype(int)
    y_max = np.ceil(np.max(tri[:, 1])).astype(int)
    
    x_min = max(0, x_min)
    x_max = min(W - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(H - 1, y_max)
    
    return x_min, x_max, y_min, y_max

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """
    H, W, C = img_float.shape
    
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    
    dc = x - x0
    dl = y - y0
    
    x1 = x0 + 1
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)
    
    a1 = img_float[y0, x0]
    a2 = img_float[y0, x1]
    a3 = img_float[y1, x0]
    a4 = img_float[y1, x1]
    
    p = (1 - dc) * (1 - dl) * a1 + dc * (1 - dl) * a2 + (1 - dc) * dl * a3 + dc * dl * a4
    
    return p

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """
    H, W, _ = A.shape
    
    frame = np.zeros_like(A, dtype=np.float32)
    
    pT = (1.0 - alfa) * pA + alfa * pB
    
    for indices_tri in triangles:
        triA = pA[indices_tri]  
        triB = pB[indices_tri]  
        triT = pT[indices_tri]  
        
        x_min, x_max, y_min, y_max = _tri_bbox(triT, W, H)
        
        for y_T in range(y_min, y_max + 1):
            for x_T in range(x_min, x_max + 1):
                pt_T = np.array([x_T, y_T], dtype=np.float32)
                
                w1, w2, w3 = _transf_baricentrica(pt_T, triT)
                
                if _check_bari(w1, w2, w3):
                    x_A = w1 * triA[0][0] + w2 * triA[1][0] + w3 * triA[2][0]
                    y_A = w1 * triA[0][1] + w2 * triA[1][1] + w3 * triA[2][1]

                    x_B = w1 * triB[0][0] + w2 * triB[1][0] + w3 * triB[2][0]
                    y_B = w1 * triB[0][1] + w2 * triB[1][1] + w3 * triB[2][1]
                    
                    cor_A = _amostra_bilinear(A, x_A, y_A)
                    cor_B = _amostra_bilinear(B, x_B, y_B)
                    
                    cor_final = (1.0 - beta) * cor_A + beta * cor_B
                    
                    frame[y_T, x_T] = cor_final

    return frame