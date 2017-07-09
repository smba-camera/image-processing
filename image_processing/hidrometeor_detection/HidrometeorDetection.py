import Queue
import numpy as np

def visibilityOfRain():
    #rain parameters
    a=1   #size of a rain drop in mm anything between 0.5 and 5 mm (for hailstone 10-40mm)
    rho=30  #rain density anything under 305 mm/hour or 38mm/min
    v=200*np.sqrt(a) #maybe velocity: 2-10 m/s (for hailstone 10-20m/s)

    #camera parameters
    f=35  #camera focal length 10-1200mm
    N=2.8 #F-Number between f/1.4–f/22
    Te = 0.5 #exposure time 1/16000-1s
    Lb = 500 #background brightness 0.0001 - 100.000
    Lr = 750 #rain brightness
    fpixel = f*0.5 #focal length in pixels
    #mixed parameters
    Zm = 2*fpixel*a #the distance after which rain becomes fog for a camera
    z = 20 #distance of rain drop from camera
    #rain visible region 0<z<R*Zm where R is a constant
    #R depends on the brightness of the scene and camera sensitivity
    #G function is unknown
    def G(f,N):
        return f/N

    vr_aprox = ((a ^ 2) * np.sqrt(rho) / np.sqrt(v))*(Lr-Lb)*((np.sqrt( G(f,N) ) ) / np.sqrt(Te) )
    print vr_aprox


# mu is the mean, sigma is the standard deviation of the kth gaussian component, X is a pixel at time t
def Gaussian_N(X, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.e ^ ((-1 / 2) * (sigma ^ (-2)) * (X - mu) ^ 2)

def movingObjectsSegmentation():
    K = 100  # number of Gaussian distributions for each pixel
    Time = 30  # time = the sum of all available pictures that will be analyzed
    total_Pixels = 640 * 480

    P = [[0 for x in range(total_Pixels)] for y in range(Time)]  # our actual images collected as a array of pixel for each timestamp
    omega = [[0 for x in range(K)] for y in range(Time)]
    mu = [[0 for x in range(K)] for y in range(Time)]
    sigma = [[0 for x in range(K)] for y in range(Time)]
    M = [[0 for x in range(K)] for y in range(Time)]
    # initialize M as such: M[k,t] = 1 if omega[k] is the first matched component, 0 otherwise
    X = [x for x in range(Time)]


    T = 0.3  # minimum prior probability of observing a background pixel
    gamma_1 = 0.001  # decay factor
    change_factor = 1 / gamma_1  # time constant which determines change
    b = 5  # b < K a subset of the total gaussian distributions used as a model of the scene background
    B = [[0 for x in range(total_Pixels)] for y in range(b)]  # this will be our background
    F = [[0 for x in range(total_Pixels)] for y in range(Time)]  # these will be our foregrounds of movement

    # for one timestamp
    for t in range(Time):
        # for one pixel
        for i in range(total_Pixels):

            temp_sum = 0
            # for one Gaussian distribution
            for k in range(K):
                temp_sum += omega[k][t] * Gaussian_N(P[i][t], mu[k][t], sigma[k][t])
                fitness_value = omega[k][t] / sigma[k][t]
                # this should be moved in a prior loop
                if k <= b and k > 0:
                    temp_sum2 = 0
                    for j in range(1, B):
                        temp_sum2 += omega[j][t]
                    B[i][k] = np.argmin(temp_sum2 > T)  # not sure if this is correct, look at page 351
                if t > 1:
                    rho = gamma_1 * Gaussian_N(X[t] | mu[k][t], omega[k][t])
                    omega[k][t] = (1 - gamma_1) * omega[k][t - 1] + gamma_1 * M[k][t]
                    mu[k][t] = (1 - rho) * mu[k][t - 1] + rho * X[t]
                    sigma[k][t] = np.sqrt((1 - rho) * np.sqr(sigma[k][t - 1]) + rho * np.sqr(X[t] - mu[k, t]))

            P[i][t] = temp_sum

            D = 3  # pixel intensity multiplier
            for x in range(b):
                isForeground = False
                if (P[i][t] > B[i][x] * D):
                    isForeground = True
            if (isForeground):
                F[i][t] = 1  # marked as foreground

def flood_fill(node,target_color,replacement_color):
    if target_color==replacement_color:
        return node
    if node.color!=target_color:
        return node
    Q = Queue.Empty
    Q.put(node)
    for N in Q:
        w = N
        e = N
        #move W to the west while the target_color==w.color
        while(target_color==w.color): w = w.west
        while(target_color==e.color): e = e.east
        for n in range(w,e):
            n.color = replacement_color
            if(n.north==target_color): Q.put(n.north)
            if(n.south==target_color): Q.put(n.south)
    return node

def photometricSelectionRule(I,t):
    n = 0  # n is the index of an image taken 0 < n < T
    c = 0  # a threshold that represents the minimum transient change in intensity caused
    isPhotoRuleSat = False
    # I is the intensity
    delta_I_1 = I[n][t] - I[n - 1][t]
    delta_I_2 = I[n][t] - I[n + 1][t]
    if delta_I_1 < 0:
        delta_I_1 = -delta_I_1
    if delta_I_2 < 0:
        delta_I_2 = -delta_I_2
    if delta_I_1 == delta_I_2 and delta_I_1 >= c:
        isPhotoRuleSat = True
    return isPhotoRuleSat

def sizeSelectionRule():
    # Section 2.4 Size Selection Rule
    # only explained in words, no math to work with
    # filter out objects that are too large or too small
    # use a flood-fill algorithm and then suppress the connected components whose size is not plausible
    # for each foreground image in F use flood-fill
    # then remove components that span too many pixels
    # make a heuristic for separating the big objects versus the small objects
    return 0

def SoftVoting():
    # TODO check soft-voting algorithm, looks like a ML classifier multiple classifiers and vote on the best

    bins = 180
    P_blobs = 300  # total number of blobs possibly the ones from the foreground
    hist = [0 for x in range(bins)]
    w = [0 for x in range(P_blobs)]  # weight
    theta = [0 for x in range(P_blobs)]  # angles from 0 to pi
    d = [0 for x in range(P_blobs)]  # uncertainty on the estimation

    # update histogram here
    temp_sum = 0
    for i in range(P_blobs):
        temp_sum = w[i] * Gaussian_N(theta, theta[i], d[i])
    hist[theta] = temp_sum

def computingHos(P_blobs):
    # parameters related to each segmented blob from section 3.2
    a = [0 for i in range(P_blobs)]  # major semiaxis
    b = [0 for i in range(P_blobs)]  # short semiaxis
    orientation_d_theta = [0 for i in range(P_blobs)]
    x0 = 0
    y0 = 0
    gravity_center = [x0, y0]  # gravity center
    theta_hos = [0 for i in range(P_blobs)]  # tilt angle
    m = [[1 for i in range(4)] for j in
         range(P_blobs)]  # 4 values for each of the P_blobs equivalent to m00,m02,m11,m20 in this order
    x = [0 for i in range(P_blobs)]
    y = [0 for i in range(P_blobs)]
    # second order moments?

    m20_sum = 1
    m11_sum = 1
    m02_sum = 1
    # the lambdas are eigenvalues of the matrix given by the 2x2 matrix of ((m20,m11)(m11,m02)) <=> ((m[3][i],m[2][i])(m[2][i],m[1][i]))
    lambda_1 = [0 for i in range(P_blobs)]
    lambda_2 = [0 for i in range(P_blobs)]
    for i in range(P_blobs):
        m20_sum += (x[i] - x0) ^ 2
        m11_sum += (x[i] - x0) * (y[i] - y0)
        m02_sum += (y[i] - y0) ^ 2

    dm = 0  # chosen empirically by performing tests in simulation (Section 3.4)

    for i in range(P_blobs):
        m[3][i] = 1 / m[0][i] * m20_sum
        m[2][i] = 1 / m[0][i] * m11_sum
        m[1][i] = 1 / m[0][i] * m02_sum
        a[i] = np.sqrt(lambda_1[i])
        b[i] = np.sqrt(lambda_2[i])
        theta_hos[i] = 1 / 2 * np.arctan(2 * m[3][i] / (m[1][i] - m[3][i]))
        orientation_d_theta[i] = np.sqrt((m[1][i] - m[3][i]) ^ 2 + 2 * (m[2][i]) ^ 2) / (
        (m[1][i] - m[3][i]) ^ 2 + 4 * (m[2][i]) ^ 2) * dm

    # The contribution of elongated ellipses to the HOS is a peaky Gaussian distribution
    # The contribution of ellipses with shapes close to disks is a flat Gaussian distribution
    # The HOS expression:
    theta_single_value = 0  # TODO investigate what this is?
    hos = 0
    for i in range(P_blobs):
        hos += a[i] / orientation_d_theta[i] * np.sqrt(2 * np.pi) * np.exp(
            (-1 / 2) * ((theta_single_value - theta_hos[i])) / orientation_d_theta[i]) ^ 2
    return hos

def evaluateRainStreaks():
    # Rain streak related parameters for evaluation of HOG and HOS
    w_s = 1  # the width in pixels
    l_s = 11  # the length in pixels
    mu_s = 55  # the mean value in degrees
    sigma_s = 10  # standard deviation in degrees

    # measure the mean value and standard deviation at the peak of the HOS on different simulated images
    mu_m = 1
    sigma_m = 1
    # relative errors on the mean and on the stan dev
    mu_f = 1  # it doesn't say what these values are
    sigma_f = 1  # it doesn't say what these values are
    epsilon_mu = (mu_f - mu_m) / mu_f
    epsilon_sigma = (sigma_f - sigma_m) / sigma_f

def modelHos():
    theta = np.pi / 2  # range between [0,PI]
    # y(theta)*d*theta the probability of observing [theta,theta+d*theta]
    capital_pi = 0  # the surface of the Gaussian distribution
    # mean mu and stan dev sigma U[0,pi](theta) = uniform distribution on the interval [0,pi]
    # HOS is modelled as a Gaussian Uniform Distribution see page 354 equation 23
    #TODO continue from here

def expectationMaximizationAlgorithm():
    #TODO call the EM alg from the ExpectationMaximization class with the right params
    return 0
