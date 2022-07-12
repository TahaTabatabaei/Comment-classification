import numpy as np
import math


def Key_Identification(vectors , k_nighboors , Data_Dimension, C_Target , g):
    vecrotsCount = len(vectors)

    data = np.empty((vecrotsCount  , Data_Dimension))
    # sakht label ha
    L_set = np.arange(vecrotsCount)
    C_Current = math.floor(vecrotsCount / g)

    for i in range(0 , vecrotsCount) :
            data[i] = vectors[i]

    if (C_Target >= C_Current):
        return L_set
    
    # mohasebe D_Original
    D_Original = np.zeros((vecrotsCount, vecrotsCount) )

    for i in range(0 , vecrotsCount ):
            for j in range(0 , vecrotsCount) :
                D_Original[i , j] = dist(i , j , data, Data_Dimension)
    

    # mohasebe R har element
    R_set = np.empty((vecrotsCount, k_nighboors + 1))
    for i in range( 0 , vecrotsCount):
        sorted = np.argsort(D_Original[i])
        for j in range(0 , k_nighboors + 1):
            R_set[i][j] = sorted[j]

    R_set = R_set.astype(int)

    # mohasebe D-Current
    D_Current = np.zeros((vecrotsCount ,vecrotsCount))

    for i in range(0 , vecrotsCount):
        for j in range(0 , i):
            # if (i == j):
            #     D_Current[i][j] = 0
            # else:
                avg_dist = 0
                for k in range(0 , k_nighboors+1):
                    for l in range(0 , k_nighboors+1):
                        # R_set[i][k] = int(R_set[i][k])

                        avg_dist += D_Original[R_set[i][k]][R_set[j][l]]

                avg_dist = avg_dist / ((k_nighboors+1)*(k_nighboors+1))
                D_Current[i][j] = avg_dist
                D_Current[j][i] = avg_dist

    while(C_Current > C_Target):
        keyPoints = findKeyPoints(n_keys=C_Current ,D_Current_set=D_Current)

    # edgham 2 cluster, ba peyda kardan cluster ba fasele minimum az cluster i
        m = len(D_Current)
        for i in range(0, m):
            min = math.inf
            min_cluster_index =0

            for j in range(0 , len(keyPoints)):
                if (D_Current[i][keyPoints[j]] < min):
                    min = D_Current[i][j]
                    min_cluster_index = j
            
            for k in range(0, len(L_set)):
                if (L_set[k] == i):
                    L_set[k] = min_cluster_index


    #  update faseleha
        p_set = [[]*vecrotsCount]

        for i in range(0, len(keyPoints)):
            for j in range(0,vecrotsCount):
                if (L_set[j] == keyPoints[i]):
                    p_set[keyPoints[i]].append(j)
                    p_set[keyPoints[i]] = list(set().union(p_set[keyPoints[i]] , R_set[j]))

        m = len(keyPoints)
        D_Current = np.zeros((m , m))

        for i in range(0, m):
             for j in range(0, i):
            #     if (i == j):
            #         D_Current[i][j] = 0
            #     else:
                    value = matrix_new_dist(p_set[i] , p_set[j] , D_Original)
                    D_Current[i][j] = value
                    D_Current[j][i] = value

        C_Current =  math.floor(C_Current/g)

    return L_set

import numba
@numba.jit(nopython=True)
def dist(i , j, data, dimension):
    
        distance = 0
        distance = np.linalg.norm(float(data[i]),float(data[j]))
        # for k in range(0 , dimension):
        #     distance += math.pow(data[i, k] - data[j, k], 2)
             
        # return math.sqrt(distance)
        return distance

def findKeyPoints(n_keys, D_Current_set):
        m = len(D_Current_set)
        min = math.inf
        s_keys = []
        key = 0

        for i in range(0, m):
            distance = 0
            for j in range(0, m):
                 distance += D_Current_set[i][j]
            avg_distance = distance/m

            if (avg_distance < min):
                min = avg_distance
                key = i
        s_keys.append(key)

        for i in range(0 , n_keys-1):
            max = -math.inf
            key = 0

            for j in range(0 , m):
                min = math.inf

                for k in range(0 , len(s_keys)):
                    if (D_Current_set[j][s_keys[k]] < min):
                        min = D_Current_set[j][s_keys[k]]

                if min > max :
                 max = min
                 key = j

            s_keys.append(key)

        return s_keys


def matrix_new_dist(p_set1 ,p_set2, Dist_Original):
    sum = 0
    for a in range( 0 , len(p_set1)) :
        for b in range( 0 , len(p_set2)):
            sum += Dist_Original[p_set1[a]][p_set2[b]]

    return  (sum / (len(p_set1) * len(p_set2)))