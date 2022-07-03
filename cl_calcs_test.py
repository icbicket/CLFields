import cl_calcs
import unittest
import numpy as np
import coord_transforms
from absl.testing import parameterized

#def DoPTest(unittest.TestCase):
#    def test_degree_of_pol_normal(self):

'''
Unittests for the following functions:
- ar_mask_calc
- degree_of_polarization
- mirror_mask3d
- mirror_outline
- angle_of_incidence
- snells_law
- brewsters_angle
- reflection_coefficients
- reflected_e
- stokes_parameters
- normalize_stokes_parameters
'''

class ARMaskCalcTest(parameterized.TestCase):
    '''
    Test angle-resolved mirror masking function
    
    '''
    @parameterized.named_parameters(
        dict(testcase_name='noslit_nohole', 
             expected=np.array([True, True, True, True,
                        False, False, False, True, 
                        False]),
             holein=False,
             slit=None,
            slit_center=0,
             orientation=0),
         dict(testcase_name='noslit_hole', 
             expected=np.array([True, True, True, True,
                        False, False, False, True, 
                        True]),
             holein=True,
             slit=None,
            slit_center=0,
             orientation=0),
         dict(testcase_name='noslit_nohole_rot90', 
             expected=np.array([True, True, True, True,
                        True, False, False, False, 
                        False]),
             holein=False,
             slit=None,
            slit_center=0,
             orientation=np.pi/2),
         dict(testcase_name='noslit_hole_rot180', 
             expected=np.array([True, True, True, True,
                        False, True, False, False, 
                        True]),
            holein=True, 
            slit=None, 
            slit_center=0,
            orientation=np.pi),
         dict(testcase_name='slit_hole', 
             expected=np.array([True, True, True, True,
                        True, False, True, True, 
                        True]),
            holein=True,
            slit=3,
            slit_center=0,
            orientation=0),
         dict(testcase_name='slit_nohole', 
             expected=np.array([True, True, True, True,
                        True, False, True, True, 
                        False]),
            holein=False,
            slit=3,
            slit_center=0,
            orientation=0),
         dict(testcase_name='slit_hole_rot90', 
             expected=np.array([True, True, True, True,
                        True, True, False, True, 
                        True]),
            holein=True,
            slit=3,
            slit_center=0,
            orientation=np.pi/2),
         dict(testcase_name='slit_nohole_rot90', 
             expected=np.array([True, True, True, True,
                        True, True, False, True, 
                        False]),
            holein=False,
            slit=3,
            slit_center=0,
            orientation=np.pi/2),
    )
    def test_holes_slits(self, expected, holein, slit, orientation, slit_center):
        '''
        test different combinations of having a hole and slit or not, and 
        orientation
        '''
        theta = np.array([
                       np.pi/2, # +y, side, in xy plane
                       np.pi/2, #+x, back, in xy plane
                       np.pi/2, # -y, side, in xy plane
                       np.pi/2, # -x, front, in xy plane
                       np.pi/2-0.2, # +y, side, slightly out of xy plane (78 deg)
                       np.pi/2-0.2, # +x, back, slightly out of xy plane
                       np.pi/2-0.2, # -y, side, slightly out of xy plane
                       np.pi/2-0.2, # -x, front, slightly out of xy plane
                       0, #hole
                       ])
        phi = np.array([
                    np.pi/2, #+y
                    0, #+x
                    3*np.pi/2, #-y
                    np.pi, #-x
                    np.pi/2, #+y
                    0, #+x
                    3*np.pi/2, #-y
                    np.pi, #-x
                    0 #hole
                    ])
        mask_calc = cl_calcs.ar_mask_calc(
            theta, phi, holein=holein, slit=slit, slit_center=slit_center, 
            orientation=orientation)
        np.testing.assert_array_equal(expected, mask_calc)

    def test_edge_of_hole(self):
        phi = np.deg2rad(np.array([5, 5, 5, 40]))
        theta = np.deg2rad(np.array([3.9, 4, 4.1, 4.1]))
        mask = np.array([True, True, False, False])
        mask_calc = cl_calcs.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            orientation=0
            )
        np.testing.assert_array_equal(mask, mask_calc)

    def test_edges_of_centred_3mm_slit(self):
        '''
        centre of slit-done
        edges of 3 mm slit-done
        edges of different size slit-done
        edges of different size, rotated slit
        edges of off-centre slit
        edges of rotated slit-done
        edges of rotate, off-centre slit
        with and without hole
        '''
        # z = sqrt(10*(2.5-x)-y^2)
        xyz = np.array([
                        [0, 1.49, 4.772829349557766], # inside slit, +y
                        [0, 1.51, 4.766539625346673], # outside slit, +y
                        [0, -1.49, 4.772829349557766], # inside slit, -y
                        [0, -1.51, 4.766539625346673], # outside slit, -y
                         [-1, 1.49, 5.725373350271578], # inside slit
                         [-1, 1.51, 5.720131117378342], # outside slit
                         [-1, -1.49, 5.725373350271578], # inside slit
                         [-1, -1.51, 5.720131117378342], # outside slit
                         [1, 1.49, 3.574898599960564], # inside slit
                         [1, 1.51, 3.5664968806939954], # outside slit
                         [1, -1.49, 3.574898599960564], # inside slit
                         [1, -1.51, 3.5664968806939954], # outside slit
                         [2.465, 0, 0.5916079783099628], # inside mirror bottom
                         [2.485, 0, 0.3872983346207433], # outside mirror bottom
                         [2.47399, 0, 0.51], # inside mirror bottom
                         [2.47599, 0, 0.49], # outside mirror bottom
                         [-2.465, 0, 0.5916079783099628], # wrong side of mirror
                         [-2.485, 0, 0.3872983346207433], # wrong side of mirror
                         [-10.74, 0, 11.50651989091402], # inside top edge mirror
                         [-10.76, 0, 11.515207336387824], # outside top edge mirror
                         [-10.74, 1.49, 11.409640660423973], # top edge, +y slit edge inside mirror
                         [-10.74, 1.51, 11.407011002011], # top edge, +y slit edge outside mirror
                         [-10.76, 1.49, 11.418401814614862], # top edge, +y slit edge outside mirror
                         [-10.76, 1.51, 11.415774174360667], # top edge, +y slit edge outside mirror
                         [-10.74, -1.49, 11.409640660423973], # top edge, -y slit edge inside mirror
                         [-10.74, -1.51, 11.407011002011], # top edge, -y slit edge outside mirror
                         [-10.76, -1.49, 11.418401814614862], # top edge, -y slit edge outside mirror
                         [-10.76, -1.51, 11.415774174360667], # top edge, -y slit edge outside mirror
                         [2.25398, 1.49, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.25198, 1.49, 0.51], # inside mirror bottom +y slit edge inside mirror
                         [2.24798, 1.51, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.24598, 1.51, 0.51], # inside mirror bottom +y slit edge outside mirror
                         [2.25398, -1.49, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.25198, -1.49, 0.51], # inside mirror bottom -y slit edge inside mirror
                         [2.24798, -1.51, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.24598, -1.51, 0.51], # inside mirror bottom -y slit edge outside mirror
                        ])
        r, theta, phi = coord_transforms.cartesian_to_spherical_coords(xyz)
        expected = np.array([
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            ])
        calculated = cl_calcs.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_offcentre_3mm_slit(self):
        '''
        edges of off-centre slit, moved 0.5 mm to the positive y
        '''
        # z = sqrt(10*(2.5-x)-y^2)
        xyz = np.array([
                        [0, 1.99, 4.586927075940928], # inside slit, +y
                        [0, 2.01, 4.578198335590105], # outside slit, +y
                        [0, -0.99, 4.901010099969189], # inside slit, -y
                        [0, -1.01, 4.896927608204965], # outside slit, -y
                         [-1, 1.99, 5.571346336389436], # inside slit
                         [-1, 2.01, 5.564162111225732], # outside slit
                         [-1, -0.99, 5.832658056152443], # inside slit
                         [-1, -1.01, 5.82922807925715], # outside slit
                         [1, 1.99, 3.3226344969015176], # inside slit
                         [1, 2.01, 3.310573968362586], # outside slit
                         [1, -0.99, 3.7443156918187332], # inside slit
                         [1, -1.01, 3.738970446526691], # outside slit
                         [-10.74, 1.99, 11.333132841363856], # top edge, +y slit edge inside mirror
                         [-10.74, 2.01, 11.329602817398323], # top edge, +y slit edge outside mirror
                         [-10.76, 1.99, 11.341953094595304], # top edge, +y slit edge outside mirror
                         [-10.76, 2.01, 11.338425816664323], # top edge, +y slit edge outside mirror
                         [-10.74, -0.99, 11.46385188320226], # top edge, -y slit edge inside mirror
                         [-10.74, -1.01, 11.462107136124667], # top edge, -y slit edge outside mirror
                         [-10.76, -0.99, 11.472571638477573], # top edge, -y slit edge outside mirror
                         [-10.76, -1.01, 11.47082821770076], # top edge, -y slit edge outside mirror
                         [2.07998, 1.99, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.07798, 1.99, 0.51], # inside mirror bottom +y slit edge inside mirror
                         [2.07198, 2.01, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.06998, 2.01, 0.51], # inside mirror bottom +y slit edge outside mirror
                         [2.37798, -0.99, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.37598, -0.99, 0.51], # inside mirror bottom -y slit edge inside mirror
                         [2.37398, -1.01, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.37198, -1.01, 0.51], # inside mirror bottom -y slit edge outside mirror
                        ])
        r, theta, phi = coord_transforms.cartesian_to_spherical_coords(xyz)
        # z = sqrt(10*(x+2.5)-y^2)
#        #z_at_y0 = np.sqrt(2.5*10-1.5**2) # z^2 = 10(x+2.5)-y^2
#        z_at_positive_y = 4.58257569495584 # np.sqrt(2.5*10-2**2) # x=0, y=2
#        #z_at_negative_y = np.sqrt(2.5*10-(-1)**2) # x=0, y=-1
#        #z_at_y_negative1_positive = np.sqrt(11) # x=-1, y=2
#        #z_at_y_negative1_negative = np.sqrt(10*(-1+2.5)-(1)**2) # x=-1, y=1
        # theta = arctan((sqrt(x^2+y^2)/z)
#        theta = np.array([
#                        0.411516846067488, # (x,y,z)=(0, 2, 4.58257569495584)
#                        0.411516846067488, # (x,y,z)=(-0.01, 2, 4.58257569495584)
#                        0.411516846067488, # (x,y,z)=(0.01, 2, 4.58257569495584)
#                        0.401516846067488, # (x,y,z)=(0.01, 2, 4.58257569495584) - 0.01 in theta
#                        0.421516846067488, # (x,y,z)=(0.01, 2, 4.58257569495584) + 0.01 in theta
#                        0.4093357035812775, # (x,y,z)=(0.01, 1.99, 4.486927075940928)
#                        0.41370006682909855, # (x,y,z)=(0.01, 2.01, 4.578198335590105)
#                        0.19135792079033082, # (x,y,z)=(0, -1, 4.898979485566356) - 0.01 in theta, -y edge
#                        0.21135792079033082, # (x,y,z)=(0, -1, 4.898979485566356) + 0.01 in theta, -y edge
#                        0.20127672852222292, # (x,y,z)=(0, -0.99, 4.901010099969189), -y edge
#                        0.2014400288989755, # (x,y,z)=(0, -1.01, 4.896927608204965), -y edge
#                        0.5931997761496288, #np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # (x,y,z)=(-1, 2, 3.3166247903554), +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+2**2)/z_at_y_negative1_positive)-0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+2**2)/z_at_y_negative1_positive)+0.01, # to spherical coordinates, +y edge, x=-1
#                        #np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1**2)/z_at_y_negative1_negative)-0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1**2)/z_at_y_negative1_negative)+0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        ])
#        xyz = np.array([[-1, 2, 3.3166247903554]])
#        print(coord_transforms.cartesian_to_spherical_coords(xyz))
#        
#        phi = np.array([
#                     np.pi/2,
#                     np.pi/2-0.01,
#                     np.pi/2+0.01,
#                     np.pi/2,
#                     np.pi/2,
#                     np.pi/2,
#                     np.pi/2,
#                    3*np.pi/2,
#                    3*np.pi/2,
#                    3*np.pi/2,
#                    3*np.pi/2,
#                    2.03444394, #-1.1071487177940904,
#                    np.arctan(2/-1),
#                    np.arctan(2/-1),
#                    #-np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    0,
#                    0,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    0,
#                    0,
#                    ])
        expected = np.array([
             False,
             True,
             False,
             True,
             False,
             True,
             False,
             True,
             False,
             True,
             False,
             True,
             False,
             True,
             True,
             True,
             False,
             True,
             True,
             True,
             True,
             False,
             True,
             True,
             True,
             False,
             True,
             True,
            ])
        calculated = cl_calcs.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0.5,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

#    def test_edges_of_centred_2p5mm_slit(self):
#        '''
#        edges of different size slit
#        '''
#        z_at_y0 = np.sqrt(2.5*10-1.25**2) # from parabola equation at x=1.5, y=0
#        z_at_y_negative1 = np.sqrt(10*(-1+2.5)-(1.25)**2) # x=1.25, y=-1
#        theta = np.array([
#                        #np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)-0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)+0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25-0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25+0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        #np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)-0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)+0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25-0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25+0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        #np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1), # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)-0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)+0.01, # to spherical coordinates, +y edge, x=-1
#                        #np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1), # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)-0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)+0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        ])
#        phi = np.array([
#                    #np.pi/2,
#                    np.pi/2-0.01,
#                    np.pi/2+0.01,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    #-3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    #np.arctan(1.25/-1),
#                    np.arctan(1.25/-1),
#                    np.arctan(1.25/-1),
#                    #-np.arctan(1.25/-1),
#                    -np.arctan(1.25/-1),
#                    -np.arctan(1.25/-1),
#                    0,
#                    0,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    0,
#                    0,
#                    ])
#        expected = np.array([
#            #False
#            False, 
#            True, 
#            False, 
#            True, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            False, 
#            True, 
#            True, 
#            True, 
#            False, 
#            True, 
#            False, 
#            False])
#        calculated = cl_calcs.ar_mask_calc(
#            theta,
#            phi,
#            holein=True,
#            slit=2.5,
#            slit_center=0,
#            orientation=0,
#            )
#        np.testing.assert_array_equal(expected, calculated)

#    def test_edges_of_centred_3mm_slit_hole_out(self):
#        '''
#        centre of slit
#        edges of 3 mm slit
#        edges of different size slit
#        edges of off-centre slit
#        edges of rotated slit
#        edges of rotate, off-centre slit
#        with and without hole
#        '''
#        z_at_y0 = np.sqrt(2.5*10-1.5**2) # from parabola equation at x=1.5, y=0
#        z_at_y_negative1 = np.sqrt(10*(-1+2.5)-(1.5)**2) # x=1.5, y=-1
#        theta = np.array([
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)-0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)+0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5-0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5+0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)-0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)+0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5-0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5+0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)-0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)+0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)-0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)+0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        ])
#        phi = np.array([
#                    np.pi/2,
#                    np.pi/2-0.01,
#                    np.pi/2+0.01,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    np.arctan(1.5/-1),
#                    np.arctan(1.5/-1),
#                    np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    0,
#                    0,
#                    np.pi,
#                    np.pi,
#                    ])
#        expected = np.array([False, False, True, False, True, False, True, False, False, True, False, True, False, False, True, False, False, True, False, True, False, True])
#        calculated = cl_calcs.ar_mask_calc(
#            theta,
#            phi,
#            holein=False,
#            slit=3,
#            slit_center=0,
#            orientation=0,
#            )
#        np.testing.assert_array_equal(expected, calculated)

#    def test_edges_of_centred_3mm_slit_rotated(self):
#        '''
#        edges of rotated slit
#        '''
#        z_at_y0 = np.sqrt(2.5*10-1.5**2) # from parabola equation at x=1.5, y=0
#        z_at_y_negative1 = np.sqrt(10*(-1+2.5)-(1.5)**2) # x=1.5, y=-1
#        theta = np.array([
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)-0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)+0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5-0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5+0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)-0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.5**2)/z_at_y0)+0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5-0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.5+0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)-0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)+0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1), # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)-0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.5**2)/z_at_y_negative1)+0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        ])
#        phi = np.array([
#                    np.pi/2,
#                    np.pi/2-0.01,
#                    np.pi/2+0.01,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    np.arctan(1.5/-1),
#                    np.arctan(1.5/-1),
#                    np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    -np.arctan(1.5/-1),
#                    0,
#                    0,
#                    np.pi,
#                    np.pi,
#                    ])-np.pi/2
#        expected = np.array([False, False, True, False, True, False, True, False, False, True, False, True, False, False, True, False, False, True, False, True, False, True])
#        calculated = cl_calcs.ar_mask_calc(
#            theta,
#            phi,
#            holein=True,
#            slit=3,
#            slit_center=0,
#            orientation=np.pi/2,
#            )
#        np.testing.assert_array_equal(expected, calculated)

#    def test_edges_of_centred_2p5mm_slit_rotated(self):
#        '''
#        edges of different size slit, rotated by pi/3
#        '''
#        z_at_y0 = np.sqrt(2.5*10-1.25**2) # from parabola equation at x=1.5, y=0
#        z_at_y_negative1 = np.sqrt(10*(-1+2.5)-(1.25)**2) # x=1.25, y=-1
#        theta = np.array([
#                        #np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)-0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)+0.01, # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25-0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25+0.01)**2)/z_at_y0), # to spherical coordinates, +y edge, x=0
#                        #np.arctan(np.sqrt(0+1.25**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)-0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+1.25**2)/z_at_y0)+0.01, # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25-0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        np.arctan(np.sqrt(0+(1.25+0.01)**2)/z_at_y0), # to spherical coordinates, -y edge, x=0
#                        #np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1), # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)-0.01, # to spherical coordinates, +y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)+0.01, # to spherical coordinates, +y edge, x=-1
#                        #np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1), # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)-0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(np.sqrt((-1)**2+1.25**2)/z_at_y_negative1)+0.01, # to spherical coordinates, -y edge, x=-1
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)-0.01,# bottom of mirror
#                        np.arctan(2.475/0.5)+0.01,# bottom of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))-0.01, # top edge of mirror
#                        np.arctan(10.75/np.sqrt(10*(10.75+2.5)))+0.01, # top edge of mirror
#                        ])
#        phi = np.array([
#                    #np.pi/2,
#                    np.pi/2-0.01,
#                    np.pi/2+0.01,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    np.pi/2,
#                    #-3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    -3*np.pi/2,
#                    #np.arctan(1.25/-1),
#                    np.arctan(1.25/-1),
#                    np.arctan(1.25/-1),
#                    #-np.arctan(1.25/-1),
#                    -np.arctan(1.25/-1),
#                    -np.arctan(1.25/-1),
#                    0,
#                    0,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    np.pi,
#                    0,
#                    0,
#                    ])-np.pi/3
#        expected = np.array([
#            #False
#            False, 
#            True, 
#            False, 
#            True, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            #False, 
#            False, 
#            True, 
#            False, 
#            True, 
#            True, 
#            True, 
#            False, 
#            True, 
#            False, 
#            False])
#        calculated = cl_calcs.ar_mask_calc(
#            theta,
#            phi,
#            holein=True,
#            slit=2.5,
#            slit_center=0,
#            orientation=np.pi/3,
#            )
#        np.testing.assert_array_equal(expected, calculated)

#    def test_near_slit(self):
#        '''
#        centre of slit
#        edges of 3 mm slit
#        edges of different size slit
#        edges of off-centre slit
#        edges of rotated slit
#        edges of rotate, off-centre slit
#        with and without hole
#        '''
#        theta = np.array([
#                        np.pi/4, #45 degrees on +x
#                        np.pi/4, #45 degrees on -x
#                        np.pi/4, #45 degrees on +y
#                        np.pi/4, #45 degrees on -y
#                        ])
#        phi = np.array([
#                    0,
#                    np.pi,
#                    np.pi/2,
#                    -np.pi/2,
#                    ])
#        expected = np.array([False, True, True, True])
#        calculated = cl_calcs.ar_mask_calc(
#            theta,
#            phi,
#            holein=True,
#            slit=3,
#            slit_center=0,
#            orientation=0,
#            )
#        np.testing.assert_array_equal(expected, calculated)

#    def test_single_value(self):
#        pass
#        
#    def test_n_by_n_array(self):
#        pass
#    
#    def test_n_by_1_array(self):
#        pass
#    
#    def test_1_by_n_array(self):
#        pass
#    
#    def test_3d_array(self):
#        pass 

#class DoPTest(unittest.TestCase):
#    def test_pol(self):
#        S0 = np.array([3,])
#        S1 = np.array([1,])
#        S2 = np.array([2,])
#        S3 = np.array([-2,])
#        dop = np.transpose(np.array([[1, np.sqrt(5)/S0[0], -2/3, -2/(3+np.sqrt(5))]]))
#        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
#        dop_calc = np.array([DoP, DoLP, DoCP, ell])
#        np.testing.assert_array_almost_equal(dop, dop_calc)


#class MirrorMask3dTest(unittest.TestCase):
#    def testMirrorMask(self):
#        pass


#class MirrorOutlineTest(unittest.TestCase):
#    def testMirrorOutline(self):
#        pass


#class AngleOfIncidenceTest(parameterized.TestCase):
#    '''
#    Test the function for calculating the angle of incidence of a wave on a 
#    surface, given the incoming wavevector and the surface normal
#    '''
#    
#    @parameterized.named_parameters(
#        ('45 degrees', 
#            np.array([0, 1, 1]), 
#            np.array([0, 0, 1]), 
#            np.array(np.pi/4),
#        ),
#        ('0 degrees',
#            np.array([0, 1, 0]),
#            np.array([0, 1, 0]),
#            np.array([0]),
#        ),
#        ('90 degrees',
#            np.array([0, 1, 0]),
#            np.array([1, 0, 0]),
#            np.array([np.pi/2]),
#        ),
#        ('45 degrees opposite to normal (+y,+z):(-z)',
#            np.array([0, 1, 1]),
#            np.array([0, 0, -1]),
#            np.array(np.pi/4),
#        ),
#        ('45 degrees opposite to normal (-y,+z):(-z)',
#            np.array([0, -1, 1]),
#            np.array([0, 0, -1]),
#            np.array(np.pi/4),
#        ),
#        ('45 degrees to normal (+y,-z):(-z)',
#            np.array([0, 1, -1]),
#            np.array([0, 0, -1]),
#            np.array(np.pi/4),
#        ),
#        ('45 degrees to normal (-y,-z):(-z)',
#            np.array([0, -1, -1]),
#            np.array([0, 0, -1]),
#            np.array(np.pi/4),
#        ),
#    )
#    def test_angle_of_incidence_single_values(self, incident, normal, expected_angle):
#        '''
#        check that the angle of incidence is calculated correctly given
#        several simple cases
#        '''
#        angle = cl_calcs.angle_of_incidence(incident, normal)
#        np.testing.assert_allclose(angle, expected_angle, atol=1e-7)

#    def test_angle_of_incidence_multi_value_array(self):
#        '''
#        check that the angle of incidence is calculated correctly given an 
#        N by 3 array
#        '''
#        incident = np.array([
#            [0, 1, 1], 
#            [0, 1, 0], 
#            [0, 1, 0], 
#            [0, 1, 1],
#            [0, -1, 1],
#            [0, 1, -1],
#            [0, -1, -1],
#            ])
#        normal = np.array([
#            [0, 1, 0], 
#            [0, 1, 0], 
#            [1, 0, 0], 
#            [0, 0, -1],
#            [0, 0, -1],
#            [0, 0, -1],
#            [0, 0, -1],
#            ])
#        expected_angles = np.array([
#            np.pi/4, 
#            0, 
#            np.pi/2, 
#            np.pi/4,
#            np.pi/4,
#            np.pi/4,
#            np.pi/4,
#            ])
#        angles = cl_calcs.angle_of_incidence(incident, normal)
#        np.testing.assert_allclose(angles, expected_angles)


#class SnellsLawTest(parameterized.TestCase):
#    '''
#    Test the Snell's law function for calculating the angle of refraction
#    '''
#    def test_angle_array(self):
#        '''
#        test the standard case in which an array of angles is used and the 
#        surface has a refractive index of 2
#        '''
#        incidence_angles = np.array([
#            0, 
#            np.pi/8, 
#            np.pi/4, 
#            np.pi/3, 
#            np.pi/2-0.1,
#            np.pi/2,
#            ])
#        n_surface = 2
#        n_environment = 1
#        expected_refraction_angles = np.array([
#            0, 
#            0.192528938, 
#            0.361367123, 
#            0.447832396, 
#            0.520716822,
#            0.523598775,
#            ])
#        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
#        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)

#    def test_single_angle(self):
#        '''
#        test a single angle value is used as input, and the surface has a 
#        refractive index of 2
#        '''
#        incidence_angles = np.pi/8
#        n_surface = 2
#        n_environment = 1
#        expected_refraction_angles = 0.192528938
#        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
#        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)

#    @parameterized.named_parameters(
#        ('critical angle', np.pi/6, np.pi/2),
#        ('below critical angle', np.pi/6-0.1, 0.965067965),
#    )
#    def test_critical_angle(self, incidence_angle, expected_refraction_angle):
#        '''
#        the second medium has a smaller refractive index than the first, test
#        the angle at which total internal reflection occurs, and a value below 
#        that
#        '''
#        n_surface = 1
#        n_environment = 2
#        refraction_angle = cl_calcs.snells_law(incidence_angle, n_surface, n_environment)
#        np.testing.assert_allclose(refraction_angle, expected_refraction_angle, atol=1e-7)
#        
#    def test_below_critical_angle(self):
#        '''
#        the second medium has a smaller refractive index than the first, test
#        an angle shallower than that at which total internal reflection occurs
#        - the result should be NaN
#        '''
#        incidence_angles = np.pi/6+0.1
#        n_surface = 1
#        n_environment = 2
#        refraction_angle = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
#        assert np.isnan(refraction_angle)


#class BrewstersAngleTest(parameterized.TestCase):
#    '''
#    Test the reflection coefficient calculation
#    '''
#    def test_single_value(self):
#        '''
#        calculate Brewster's angle for n1=1, n2=2
#        '''
#        n_surface = 2
#        n_environment = 1
#        expected_brewsters = 1.107148718
#        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
#        np.testing.assert_allclose(brewsters, expected_brewsters)
#    
#    def test_value_array(self):
#        '''
#        calculate Brewster's angle for n1=1, n2=2, using arrays for refractive
#            indices
#        '''
#        n_surface = np.array([2, 3])
#        n_environment = np.array([1, 2])
#        expected_brewsters = np.array([1.107148718, 0.982793723])
#        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
#        np.testing.assert_allclose(brewsters, expected_brewsters)


#class ReflectionCoefficientsTest(parameterized.TestCase):
#    '''
#    Test the reflection coefficient calculation
#    '''
#    def test_brewsters_angle(self):
#        '''
#        the parallel reflection coefficient should be 0 at Brewster's angle
#        '''
#        n_surface = 2
#        n_environment = 1
#        incidence_angle = 1.107148718
#        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
#        expected_r_p = 0
#        np.testing.assert_allclose(r_p, expected_r_p, atol=1e-7, equal_nan=False)

#    def test_normal_incidence(self):
#        '''
#        the reflection coefficients should be equal at normal incidence
#        '''
#        n_surface = 2
#        n_environment = 1
#        incidence_angle = 0
#        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
#        np.testing.assert_allclose(r_p, r_s, atol=1e-7)
#    
#    def test_single_value(self):
#        '''
#        test the reflection coefficient calculation with a single input angle
#        '''
#        n_surface = 2
#        n_environment = 1
#        incidence_angle = 1.
#        r = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
#        expected_r_s = np.array([-0.54108004])
#        expected_r_p = np.array([0.087243335])
#        np.testing.assert_allclose(r, (expected_r_s, expected_r_p), atol=1e-7)
#    
#    def test_array_of_values(self):
#        '''
#        test the reflection coefficient calculation with a 1D numpy array input
#            for incidence angle
#        '''
#        n_surface = 2
#        n_environment = 1
#        incidence_angle = np.array([1., np.pi/4])
#        r = np.array(cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment))
#        expected_r = np.transpose(np.array([[-0.54108004, 0.087243335], [-0.451416229, 0.203776612]]))
#        np.testing.assert_allclose(r, expected_r, atol=1e-7)


#class ReflectedETest(parameterized.TestCase):
#    '''
#    Test the calculation of the reflected electric field
#    '''
#    
#    @parameterized.named_parameters(
#        ('s-polarized', np.array([0, 0, 1]), np.array([0, 0, -0.451416229]), np.array([0, 0, 0])),
#        ('p-polarized', np.array([1, -1, 0]), np.array([0, 0, 0]), 0.203776612*np.array([1, -1, 0])),
#        ('mixed-polarized', np.array([1, -1, 1]), np.array([0, 0, -0.451416229]), 0.203776612*np.array([1, -1, 0])),
#        ('3 by 3 array', np.array([[0, 0, 1], [1, -1, 0], [1, -1, 1]]), np.array([[0, 0, -0.451416229], [0, 0, 0], [0, 0, -0.451416229]]), 0.203776612 * np.array([[0, 0, 0], [1, -1, 0], [1, -1, 0]]))
#    )
#    def test_e_polarization_state(self, incident_e, expected_e_s, expected_e_p):
#        '''
#        An input electric field of various polarization states
#        '''
#        normal = np.array([1, 0, 0])
#        incident_direction = np.array([1, 1, 0])
#        n_surface = 2
#        n_environment = 1
#        e_s, e_p = cl_calcs.reflected_e(
#            incident_direction,
#            incident_e,
#            normal,
#            n_surface,
#            n_environment
#            )
#        np.testing.assert_allclose(e_s, expected_e_s)
#        np.testing.assert_allclose(e_p, expected_e_p)


#    @parameterized.named_parameters(
#        ('s-polarized', np.array([0, 0, 1]), np.array([0, 0, -0.451416229]), np.array([0, 0, 0])),
#        ('p-polarized', np.array([1, -1, 0]), np.array([0, 0, 0]), 0.203776612*np.array([1, -1, 0])),
#        ('mixed-polarized', np.array([1, -1, 1]), np.array([0, 0, -0.451416229]), 0.203776612*np.array([1, -1, 0])),
#    )
#    def test_e_polarization_state_negative_normal(self, incident_e, expected_e_s, expected_e_p):
#        '''
#        An input electric field of various polarization states
#        '''
#        normal = np.array([-1, 0, 0])
#        incident_direction = np.array([1, 1, 0])
#        n_surface = 2
#        n_environment = 1
#        e_s, e_p = cl_calcs.reflected_e(
#            incident_direction, 
#            incident_e, 
#            normal, 
#            n_surface, 
#            n_environment
#            )
#        np.testing.assert_allclose(e_s, expected_e_s)
#        np.testing.assert_allclose(e_p, expected_e_p)


#class StokesParametersTest(unittest.TestCase):
#    def testS1LinearPolarized(self):
#        '''
#        check inputs which should result in x or y polarization
#        '''
#        E1 = np.array([1, -1, 0, 0])
#        E2 = np.array([0, 0, 1, -1])
#        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
#        S_calc = np.array([S0, S1, S2, S3])
#        stokes = np.transpose(np.array([
#            [1, 1, 0, 0], 
#            [1, 1, 0, 0], 
#            [1, -1, 0, 0], 
#            [1, -1, 0, 0]
#            ]))
#        np.testing.assert_array_equal(S_calc, stokes)

#    def testS3CircularPolarized(self):
#        E1 = np.array([1 + 1j, 1 + 0.5j, 1 - 1j])
#        E2 = np.array([1 - 1j, 0.5 - 1j, 1 + 1j])
#        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
#        S_calc = np.array([S0, S1, S2, S3])
#        stokes = np.transpose(np.array([[4, 0, 0, 4], [2.5, 0, 0, 2.5], [4, 0, 0, -4]]))
#        np.testing.assert_array_equal(stokes, S_calc)

#    def test_stokes_angle_polarized(self):
#        E1 = 1 + 1j
#        E2 = 1 - 0j
#        stokes = np.array([3, 1, 2, 2])
#        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
#        S_calc = np.array([S0, S1, S2, S3])
#        np.testing.assert_array_equal(stokes, S_calc)

#    def testS2LinearPolarized(self):
#        E1 = np.array([1, 1, -1, -1])
#        E2 = np.array([1, -1, 1, -1])
#        stokes = np.transpose(np.array([
#            [2, 0, 2, 0], 
#            [2, 0, -2, 0], 
#            [2, 0, -2, 0],
#            [2, 0, 2, 0],
#            ]))
#        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
#        S_calc = np.array([S0, S1, S2, S3])
#        np.testing.assert_array_equal(stokes, S_calc)


#class NormalizeStokesParametersTest(unittest.TestCase):
#    def test_normal(self):
#        '''
#        Test a normal Stokes vector
#        '''
#        S0 = np.array([3])
#        S1 = np.array([1])
#        S2 = np.array([2])
#        S3 = np.array([-2])
#        s1_expected, s2_expected, s3_expected = np.array([1/3, 2/3, -2/3])
#        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
#        np.testing.assert_array_almost_equal(
#            np.array([s1, s2, s3]),
#            np.array([[s1_expected], [s2_expected], [s3_expected]])
#            )
#    
#    def test_most_zeros(self):
#        '''
#        Test a Stokes vector where all but S2 are 0
#        '''
#        S0 = np.array([2])
#        S1 = np.array([0])
#        S2 = np.array([2])
#        S3 = np.array([0])
#        s1_expected, s2_expected, s3_expected = np.array([0, 1, 0])
#        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
#        np.testing.assert_array_almost_equal(
#            np.array([s1, s2, s3]),
#            np.array([[s1_expected], [s2_expected], [s3_expected]])
#            )

#    def test_zero_stokes(self):
#        '''
#        Test a Stokes vector with all 0 components
#        '''
#        S0 = np.array([0])
#        S1 = np.array([0])
#        S2 = np.array([0])
#        S3 = np.array([0])
#        s1_expected, s2_expected, s3_expected = np.array([0, 0, 0])
#        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
#        np.testing.assert_array_almost_equal(
#            np.array([s1, s2, s3]),
#            np.array([[s1_expected], [s2_expected], [s3_expected]])
#            )


if __name__ == '__main__':
    unittest.main()
