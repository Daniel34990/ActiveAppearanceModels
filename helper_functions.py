import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.transform import warp

def AffineRegistrationLandmarks(x,y):

    '''
    Inputs:
               x: [M,2] array containing the M 2-dim source landmarks
               y: [M,2] array containing the M 2-dim target landmarks

    Outputs:
               xp: [M,2] array containing the M 2-dim aligned source landmarks
               T: [3,3] transformation matrix

    '''

    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise NameError('data should have the same dimensions')

    if x.shape[1] != 2:
        raise NameError('This code works only for 2 dimennsional data')

    M=x.shape[0]

    Y = y.T.flatten()
    X = np.zeros((2*M,6))
    X[:M,:2] = x
    X[M:,3:5] = x
    X[:M,2] = 1
    X[M:,-1] = 1
    
    T_line = np.linalg.inv(X.T @ X) @ X.T @ Y
    T = np.zeros((3,3))
    T[0,:2] = T_line[:2]
    T[1,:2] = T_line[3:5]
    T[0,2] = T_line[2]
    T[1,2] = T_line[5]
    T[2,2] = 1

    xp_line = X @ T_line
    xp = np.zeros_like(x)
    xp[:,0] = xp_line[:M]
    xp[:,1] = xp_line[M:]
    
    return xp,T



def procrustes_align(x,y,mode='best',verbose=1):

    """
    Inputs:
               X: [M,2] array containing the M 2-dim source landmarks
               Y: [M,2] matrix containing the M 2-dim target landmarks
               mode: 'rotation' to have only rotation, 'reflection' to
                   have only reflection and 'best' to have the one decided by the
                   data depending on det(U*V')
               verbose: 1 to have explanations and 0 otherwise

    Outputs:
               Xp: [M,2] array containing the aligned source landmarks
               s: uniform scaling
               R: rotation or reflection matrix
               t: translation vector
               SSR: sum of squared of residuals
               ratio_SSR: ratio of SSR with respect to the initial SSR

    """

    if mode.lower()!='best' and mode.lower()!='rotation' and mode.lower()!='reflection':
        raise NameError('Error ! mode should be equal to best, rotation or reflection')

    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise NameError('data should have the same dimensions')

    if x.shape[1] != 2:
        raise NameError('This code works only for 2 dimennsional data')

    M=x.shape[0]

    # Center data
    X_c = x - np.mean(x,axis=0)
    Y_c = y - np.mean(y,axis=0)

    # Optimal parameters (hint: use np.linalg.svd)
    U, D, Vt = np.linalg.svd(X_c.T @ Y_c)
    D = np.diag(D)

    if mode.lower() == 'rotation':
        if np.absolute(np.linalg.det(U @ Vt)-1)<1e-5: # det(R)==1
            if verbose == 1:
                print('The best R is a rotation. Computing rotation.')
            S=np.eye(2)
        elif np.absolute(np.linalg.det(U @ Vt)+1)<1e-5: # det(R)== -1
            if verbose == 1:
                print('The best R is a reflection but a rotation is computed as requested.')
            S=np.array([[1, 0], [0, np.linalg.det(U@Vt)]]) # to have det(U*V')=1
        else:
            raise NameError('Error ! U*Vt should be an orthogonal matrix')
    elif mode.lower() == 'reflection':
        if np.absolute(np.linalg.det(U @ Vt)-1)<1e-5: # det(R)==1
            if verbose == 1:
                print('The best R is a rotation but a reflection is computed as requested.')
            S=np.array([[1, 0], [0, -np.linalg.det(U@Vt)]]) # to have det(U*V')=-1
        elif np.absolute(np.linalg.det(U @ Vt)+1)<1e-5: # det(R)== -1
            if verbose == 1:
                print('The best R is a reflection. Computing reflection.')
            S=np.eye(2)
        else:
            raise NameError('Error ! U*Vt should be an orthogonal matrix')
    elif mode.lower() == 'best':
        if np.absolute(np.linalg.det(U @ Vt)-1)<1e-5: # det(R)==1
            if verbose == 1:
                print('The best R is a rotation. Computing rotation.')
            S=np.eye(2)
        elif np.absolute(np.linalg.det(U @ Vt)+1)<1e-5: # det(R)== -1
            if verbose == 1:
                print('The best R is a reflection. Computing reflection.')
            S=np.eye(2)
        else:
            raise NameError('Error ! U*Vt should be an orthogonal matrix')

    R=U @ S @ Vt
    s=np.trace(S @ D)/(np.trace(X_c.T @ X_c))

    if mode.lower() == 'rotation':
        if np.absolute(np.linalg.det(R)-1)>1e-5:
            raise NameError('Error ! there is a problem...')
    if mode.lower() == 'reflection':
        if np.absolute(np.linalg.det(R)+1)>1e-5:
            raise NameError('Error ! there is a problem...')

    t = np.mean(y,axis=0) - s* np.mean(x,axis=0) @ R
    # print(t.shape)
    xp = (s * x @ R) + t

    # Procrustes residuals
    SSR = np.sum(np.power((y-xp),2))

    # Ratio with initial residual
    SSR0 = np.sum(np.power((y-x),2))
    ratioSSR = SSR*100/SSR0
    # print(SSR)
    # print(SSR0)
    # print(ratioSSR)

    return xp, s, R, t, SSR, ratioSSR

def nearestNeighboutInterp(pM,I,coords=None):
    ''' 
    Nearest Neighbout interpolation
        
    Inputs: 
        pM: 2D point defining the coordinates to interpolate
        I: image used for interpolation
        coords: coordinates of the image. If None, the coordinates of a pixel
                are automatically its row and column position
                    
    Output:
        value: interpolated value at pM
    ''' 
    
    if coords is None:
        # row and column of pM
        r = int(round(pM[0]))
        c = int(round(pM[1]))
        
        # check if r and c are within the domain of I (I.shape)
        if (r>=0) and (r<I.shape[0]) and (c>=0) and (c<I.shape[1]):
            value = I[r,c]
        else:
            value=0
        
    else:
        raise ValueError("Error ! Still not implemented")
        value=0
        
    return value

def inverse_warping(Is,T,H,W):
    Ism = warp(
        Is, 
        inverse_map=LA.inv(T),  # Transformation inverse (cible → source)
        output_shape=(H,W),
        mode='constant',  # Remplissage avec des zéros hors de l'image
        cval=0, 
        preserve_range=True  # Conserve les valeurs originales (utile pour les images uint8)
    ) 
    return Ism

def applyTransformation(T, points=None, coords=None):
    ''' 
    Apply geometric transformation to points or image coordinates.
    Transformation is defined by a 3x3 matrix
        
    Inputs: 
        points: Nx2 Numpy array of points 
        coordinates: 2xNxM Numpy array of image coordinates
        T: 3x3 matrix trasformation
            
    Output:
        pm: Nx2 points after transformation
        cm: 2xNxM image coordinates after transformation
    ''' 
    if points is None and coords is None:
        raise ValueError("Error ! You should provide points and/or coords")
    
    if points is not None:    
        N,d = points.shape
        if d != 2 and N==2:
            print('WARNING ! points should be an array of dimension Nx2'+
                  ' Transposing the array')
            points=points.T
            N,d = points.shape
            
        if d != 2:
            raise ValueError("Error ! Function works only with 2D points")
            
        # Transform points into homogeneous coordinates (adding one...)
        homogenous_points = np.zeros((N,3))
        homogenous_points[:,:2] = points
        homogenous_points[:,2] = 1
        
        # Apply transformation
        pm = homogenous_points @ T.T 
        
        # If homography, ...
        if T[2][0]!=0:
            pm[:,0]/=pm[:,2]
            pm[:,1]/=pm[:,2]

        pm = pm[:,:2]
    else:
        pm=None
        
    if coords is not None:
        d,N,M = coords.shape
        
        if d != 2:
            raise ValueError("Error ! Function works only with 2D coordinates")
        
        coords = coords.reshape((2,N*M)) # reshape coordinates as list of points
        # Transform points into homogeneous coordinates (adding one...)
        homogeneous_coords = np.zeros((3,N*M))
        homogeneous_coords[:2,:] = coords
        homogeneous_coords[2,:] = 1
        
        # Apply transformation
        cm = T @ homogeneous_coords
        
        # If homography, ...
        if T[2][0]!=0:
            p[0,:]/=p[2,:]
            p[1,:]/=p[2,:]
        cm = cm[:2,:]
        cm = cm.reshape((2,N,M))
        
    else:
        cm = None
                
    return pm,cm

def generalized_procrustes_analysis(X,tau=1e-5,tangent=1):
    """
    Inputs:
            X: [N,2M] array containing N configurations of 2D landmarks.
               Each configuration has M landmarks
            tau: parameter for the stopping criteria (please refer to the slides
                 of the course)
            tangent: if set to 1, data will be projected onto the tangent space
    
    Outputs:
            Xm1: [M,2] array containing the landmarks of the average configuration
            Xcp: [N,2M] array containing the aligned landmarks onto Xm1
    
    """
    
    if X.shape[1] % 2 != 0:
        raise NameError('This code works only for 2 dimennsional data')
    
    # Parameters
    N,M=X.shape
    dim=2
    M=int(M/dim)
    
    # Plot original data
    plt.figure()
    for i in range(0,N):
      landmark=X[i]
      x=landmark[::2]
      y=landmark[1::2]
      plt.scatter(x, y, c='r')
    plt.gca().invert_yaxis()
    plt.title('Original landmarks')
    
    
    # Center each configuration
    Xc=np.zeros((N,M*dim))
    for i in range(0,N):
        landmark=X[i]
        x=landmark[::2]
        y=landmark[1::2]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        Xc[i,::2] = x - x_mean
        Xc[i,1::2] = y - y_mean
        
    
    # Compute first average configuration
    Xm0 = np.mean(Xc,axis=0)
    print(Xm0.shape)
    
    # Plot configurations and first average
    plt.figure()
    for i in range(0,N):
      landmark=Xc[i]
      x=landmark[::2]
      y=landmark[1::2]
      plt.scatter(x, y, c='r')
    plt.scatter(Xm0[::2],Xm0[1::2],c='g',label='average')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().invert_yaxis()
    plt.title('Centered data with first average')
    
    # Scale to unit size average
    Xm0 = Xm0.reshape((M,dim))
    C = np.eye(M) - 1/M*(np.ones((M,M)))
    norm = np.sqrt(np.sum((C @ Xm0)**2))
    Xm0 = Xm0/norm
    
    # Procrustes alignement of all configurations to the average Xm0
    Xcp=np.zeros((N,M*dim))
    for i in range(0,N):
        landmark=Xc[i]
        xs = np.reshape(landmark,(M,dim))
        xp, s, R, t, SSR, ratioSSR = procrustes_align(xs,Xm0,verbose=0)
        Xcp[i] = np.reshape(xp,M*dim)
    
    # Reshape average as vector
    Xm0 = Xm0.reshape((M*dim))
    
    # Plot configurations and average
    plt.figure()
    for i in range(0,N):
      landmark=Xcp[i]
      x=landmark[::2]
      y=landmark[1::2]
      plt.scatter(x, y, c='r')
    plt.scatter(Xm0[::2],Xm0[1::2],c='g',label='average')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Aligned data to normalized initial mean')
    
    # Tangent space projection
    if tangent==1:
        # landmarks after tangent space projection
        Xcpt=np.zeros((N,M*dim))
        # vector measuring the difference before/after projection
        diff = np.zeros((N,1))
        mean_square = np.sum(Xm0**2)
        
        for i in range(0,N):
            alpha = mean_square/(np.sum(Xcp[i]*Xm0))
            Xcpt[i]=alpha*Xcp[i]
            diff[i]=np.sum((Xcpt[i]-Xm0)**2)

        # we look for the subject with the maximum difference before/after projection
        ind=np.argmax(diff)
        
        # Plot configurations and first average
        plt.figure()
        l=Xcp[ind]
        lt=Xcpt[ind]
        plt.scatter(l[::2], l[1::2], c='r', label='before projection')
        plt.scatter(lt[::2], lt[1::2], c='b', label='after projection')
        plt.gca().invert_yaxis()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Subject with maximum variation')
        
        plt.figure()
        for i in range(0,N):
          landmark=Xcp[i]
          x=landmark[::2]
          y=landmark[1::2]
          if i==ind:
              plt.scatter(x, y, c='b',label='Subject with max distortion', zorder=10)
          else:
              plt.scatter(x, y, c='r')
        
        plt.scatter(Xm0[::2],Xm0[1::2],c='g',label='average', zorder=5)
        plt.gca().invert_yaxis()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Subjects before projection')
        
        Xcp=Xcpt
    
    # Re-estimate average configuration
    Xm1=np.mean(Xcp,axis=0)
    
    # Procrustes alignement of Xm1 to Xm0
    Xm1, s, R, t, SSR, ratioSSR = procrustes_align(np.reshape(Xm1,(M,dim)) ,np.reshape(Xm0,(M,dim)),'best',0)
    
    # Scale to unit size new average Xm1
    C = np.eye(M) - 1/M*(np.ones((M,M)))
    norm = np.sqrt(np.sum((C @ Xm1)**2))
    Xm1 = Xm1/norm
    
    # Reshape average as vector
    Xm1=Xm1.reshape(M*dim)
    
    # Plot configurations and new average
    plt.figure()
    for i in range(0,N):
      landmark=Xcp[i]
      x=landmark[::2]
      y=landmark[1::2]
      plt.scatter(x, y, c='r')
    plt.scatter(Xm1[::2],Xm1[1::2],c='g',label='average')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Aligned data to new mean')
    
    
    itera=1
    while np.sqrt(np.dot((Xm0-Xm1),(Xm0-Xm1))) > tau:
        print('Iter number %d , Error: %f' % (itera, np.sqrt(np.dot((Xm0-Xm1),(Xm0-Xm1))) ) )
        itera=itera+1
        
        # Update Xm0 to Xm1
        Xm0=Xm1
        
        # Procrustes alignement of all configurations to the average Xm0
        for i in range(0,N):
          temp=np.reshape(Xc[i],(M,dim))
          xp, s, R, t, SSR, ratioSSR = procrustes_align(temp,np.reshape(Xm0,(M,dim)),'best',0)
          Xcp[i]=np.reshape(xp,(M*dim))
        
        # Tangent space projection
        if tangent==1:
          for i in range(0,N):
              alpha = mean_square/(np.sum(Xcp[i]*Xm0))
              Xcp[i]=alpha*Xcp[i]
        
        # Re-estimate average configuration
        Xm1=np.mean(Xcp,axis=0)
        
        # Procrustes alignement of Xm1 to Xm0
        Xm1, s, R, t, SSR, ratioSSR = procrustes_align(np.reshape(Xm1,(M,dim)),np.reshape(Xm0,(M,dim)),'best',0)
        
        # Scale to unit size new average Xm1
        C = np.eye(M) - 1/M*(np.ones((M,M)))
        norm = np.sqrt(np.sum((C @ Xm1)**2))
        Xm1 = Xm1/norm
        
        # Reshape average as vector
        Xm1=np.reshape(Xm1,(M*dim))
        
        # Plot configurations and new average
        plt.figure()
        for i in range(0,N):
          landmark=Xcp[i]
          x=landmark[::2]
          y=landmark[1::2]
          plt.scatter(x, y, c='r')
        plt.scatter(Xm1[::2],Xm1[1::2],c='g',label='average')
        plt.gca().invert_yaxis()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('GPA results after iter %i' % itera)
        

    return Xcp, Xm1

def rescale_to_original(landmark, H, W):
    min_x, max_x = np.min(landmark[:, 0]), np.max(landmark[:, 0])
    min_y, max_y = np.min(landmark[:, 1]), np.max(landmark[:, 1])
    
    landmarks_mapped = np.zeros_like(landmark)
    landmarks_mapped[:, 0] = ((landmark[:, 0] - min_x) / (max_x - min_x)) * W
    landmarks_mapped[:, 1] = ((landmark[:, 1] - min_y) / (max_y - min_y)) * H
    
    landmarks_mapped = landmarks_mapped.astype(int)

    return landmarks_mapped