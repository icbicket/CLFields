from absl.testing import parameterized
import numpy as np
import  mirror_operations as miop
import unittest
import coord_transforms

'''
parabola position:
- points to positive x
- position at origin
- (-2.5, 0, 0) on parabola
- (+x, 0, 0) doesn't exist - is NaN
- 

parabola normals:
- at r = 0, unit normal should point along x
- all normals point in the same direction
- all normals should be unit vectors
'''

class MirrorXYZTest(parameterized.TestCase):
    '''
    test the calculation of xyz coordinates on the mirror surface
    '''
    

    def testCoordinates(self):
        xyz = np.array([
            [0, 1.49, 4.772829349557766],
            [0, 1.51, 4.766539625346673],
            [0, -1.49, 4.772829349557766],
            [0, -1.51, 4.766539625346673],
             [-1, 1.49, 5.725373350271578],
             [-1, 1.51, 5.720131117378342],
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
        calculated_x, calculated_y, calculated_z, calculated_c = miop.mirror_xyz(theta, phi)
        np.testing.assert_allclose(xyz[:, 0], calculated_x, atol=1e-7)
        np.testing.assert_allclose(xyz[:, 1], calculated_y, atol=1e-7)
        np.testing.assert_allclose(xyz[:, 2], calculated_z, atol=1e-7)

    def testCoordinatesThetaPhi(self):
        '''
        Expected values from Matlab functions from AMOLF
        Check the calculated value of c based on results from Matlab
        '''
        theta_phi = np.array([
            [np.pi/4, 0],
            [np.pi/4, 0.1],
            [np.pi/4, -0.1],
            [np.pi/2, 0.1],
            [np.pi/3, 0.1],
            [np.pi/3, np.pi],
            [np.pi/3, -np.pi],
            [np.pi/3, np.pi/2],
            [np.pi/3, -np.pi/2],
            [0, np.pi/2],
            [0, np.pi/6],
            [0.1, np.pi/6],
            [np.pi/6, 3*np.pi/2],
            [np.pi/6, 1],
            [np.pi/6, 2],
            [np.pi/6, -1],
            [np.pi/6, -2],
            [np.pi/2, 1],
            [3, 1],
            [3, -1],
            ])
        calculated_x, calculated_y, calculated_z, calculated_c = miop.mirror_xyz(
            theta_phi[:, 0], 
            theta_phi[:, 1], 
            )
        expected_c = np.array([
            2.928932188134524,
            2.935005720200909,
            2.935005720200909,
            2.506260431442856,
            2.685718965062429,
            37.320508075688757,
            37.320508075688757,
            5,
            5,
            5,
            5,
            4.602109547346796,
            5,
            3.936539354745236,
            6.313716593651671,
            3.936539354745236,
            6.313716593651671,
            3.246116026023812,
            4.645771682699788,
            4.645771682699788,
            ])
        np.testing.assert_allclose(calculated_c, expected_c, atol=1e-7)

class MirrorMask3dTest(unittest.TestCase):
    '''
    ar_mask_calc has been tested: test that output size is as expected given:
    - 1 element array
    - n element array
    - n by 1 array
    - 1 by n array
    - n by m array
    - n by m by p array
    '''
    def test_1_element_array(self):
        '''
        Shape should be 1 by 3 by 3, full of False
        '''
        theta = np.array([np.pi/4])
        phi = np.array([0.1])
        expected = np.array([False, False, False, False, False, False, False, False, False]).reshape(1,3,3)
        calculated_mirror_3d = miop.mirror_mask3d(
            theta, 
            phi, 
            holein=True, slit=None, orientation=0,
            )
        self.assertFalse(np.any(calculated_mirror_3d))
        self.assertEqual((1,3,3), np.shape(calculated_mirror_3d))

    def test_n_element_array(self):
        '''
        Shape should be n by 3 by 3
        n values are repeated along the other axes
        '''
        theta = np.array([np.pi/4, 0, 0, np.pi/3])
        phi = np.array([0, np.pi/6, np.pi/2, 0.1])
        expected = np.array([
            [[False, False, False], [False, False, False], [False, False, False]], 
            [[True, True, True], [True, True, True], [True, True, True]], 
            [[True, True, True], [True, True, True], [True, True, True]], 
            [[False, False, False], [False, False, False], [False, False, False]]
            ])
        calculated_mirror_3d = miop.mirror_mask3d(
            theta, 
            phi, 
            holein=True, slit=None, orientation=0,
            )
        self.assertEqual((4,3,3), np.shape(calculated_mirror_3d))
        np.testing.assert_array_equal(expected, calculated_mirror_3d)

    def test_n_by_1_array(self):
        '''
        Shape should be n by 1 by 3 by 3
        n values are repeated along the other axes
        '''
        theta = np.array([np.pi/4, 0, 0, np.pi/3]).reshape((4, 1))
        phi = np.array([0, np.pi/6, np.pi/2, 0.1]).reshape((4, 1))
        expected = np.array([
            [[[False, False, False], [False, False, False], [False, False, False]]], 
            [[[True, True, True], [True, True, True], [True, True, True]]], 
            [[[True, True, True], [True, True, True], [True, True, True]]], 
            [[[False, False, False], [False, False, False], [False, False, False]]]
            ])
        calculated_mirror_3d = miop.mirror_mask3d(
            theta, 
            phi, 
            holein=True, slit=None, orientation=0,
            )
        self.assertEqual((4, 1, 3, 3), np.shape(calculated_mirror_3d))
        np.testing.assert_array_equal(expected, calculated_mirror_3d)

    def test_1_by_n_array(self):
        '''
        Shape should be 1 by n by 3 by 3
        n values are repeated along the other axes
        '''
        theta = np.array([np.pi/4, 0, 0, np.pi/3]).reshape((1, 4))
        phi = np.array([0, np.pi/6, np.pi/2, 0.1]).reshape((1, 4))
        expected = np.array([[
            [[False, False, False], [False, False, False], [False, False, False]], 
            [[True, True, True], [True, True, True], [True, True, True]], 
            [[True, True, True], [True, True, True], [True, True, True]], 
            [[False, False, False], [False, False, False], [False, False, False]]
            ]])
        calculated_mirror_3d = miop.mirror_mask3d(
            theta, 
            phi, 
            holein=True, slit=None, orientation=0,
            )
        self.assertEqual((1, 4, 3, 3), np.shape(calculated_mirror_3d))
        np.testing.assert_array_equal(expected, calculated_mirror_3d)

    def test_n_by_m_by_p_array(self):
        '''
        Shape should be n by m by p by 3 by 3
        n by m by p values are repeated along the other axes
        '''
        theta = np.array([[[np.pi/4, 0], [0, 0.1]], [[3, np.pi/3], [np.pi/2, 3]]])  
            #[[[False, True], [True, False]], [[True, False], [True, True]]]
        phi = np.array([[[0, np.pi/6], [np.pi/2, np.pi/6]], [[-1, 0.1], [0.1, 1]]])
        expected = np.array(
            [
                [
                    [
                        [[False, False, False], [False, False, False], [False, False, False]], 
                        [[True, True, True], [True, True, True], [True, True, True]]
                    ],
                    [
                        [[True, True, True], [True, True, True], [True, True, True]], 
                        [[False, False, False], [False, False, False], [False, False, False]]
                    ],
                ],
                [
                    [
                        [[True, True, True], [True, True, True], [True, True, True]], 
                        [[False, False, False], [False, False, False], [False, False, False]]
                    ],
                    [
                        [[True, True, True], [True, True, True], [True, True, True]],
                        [[True, True, True], [True, True, True], [True, True, True]]
                    ]
                ]
            ]
            )
        calculated_mirror_3d = miop.mirror_mask3d(
            theta, 
            phi, 
            holein=True, slit=None, orientation=0,
            )
        self.assertEqual((2, 2, 2, 3, 3), np.shape(calculated_mirror_3d))
        np.testing.assert_array_equal(expected, calculated_mirror_3d)



class MirrorOutlineTest(unittest.TestCase):
    '''
    is the right shape
    max of theta and phi is as expected for:
     - no slit
     - centred 3 mm slit
     - off-centre 3 mm slit
     - centred 2.5 mm slit
     - off-centre 2.5 mm slit
     - hole
     - no hole
     - orientation rotated
     - way off centre slit? - fail?
    '''
    def test_no_hole(self):
        '''
        the hole should not be present, theta and phi for the hole should be None
        '''
        calculated_maxthph, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 1000), holein=False, slit=None, slit_center=0, orientation=0
            )
        self.assertTrue(np.all(calculated_hole == None))
        self.assertTrue(calculated_hole[:, 1] == None)
        self.assertTrue(calculated_hole[:, 0] == None)
    
    def test_hole(self):
        '''
        the hole should be present, all theta should be about 4, phi should go from 0 to 2*pi
        '''
        calculated_maxthph, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 1000), holein=True, slit=None, slit_center=0, orientation=0
            )
        np.testing.assert_allclose(calculated_hole[:, 0], 4.0)
        self.assertAlmostEqual(calculated_hole[0, 1], 0)
        self.assertAlmostEqual(calculated_hole[-1, 1], np.pi*2)

    def test_no_slit_high_n(self):
        '''
        Check some mirror limits with high n and with no slit inserted
        '''
        calculated_theta_phi, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 1000), holein=True, slit=None, slit_center=0, orientation=0
            )
#             [2.465, 0, 0.5916079783099628], # inside mirror bottom
#             [2.485, 0, 0.3872983346207433], # outside mirror bottom
#             [2.47399, 0, 0.51], # inside mirror bottom
#             [2.47599, 0, 0.49], # outside mirror bottom
#             [-10.74, 0, 11.50651989091402], # inside top edge mirror
#             [-10.76, 0, 11.515207336387824], # outside top edge mirror
        self.assertAlmostEqual(calculated_hole[0, 1], 0)
        self.assertAlmostEqual(calculated_hole[-1, 1], np.pi*2)
        self.assertTrue(np.all(calculated_theta_phi[:, 0] <= 1.5390609532664388+1e-5))
        self.assertTrue(np.shape(calculated_hole) == (1000, 2))
        self.assertTrue(np.shape(calculated_theta_phi) == (1000, 2))
    
    def test_no_slit_n10(self):
        '''
        Check some mirror limits with n=10 and with no slit inserted
        '''
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 10), holein=True, slit=None, slit_center=0, orientation=0
            )
        expected_hole = np.array([
            [4, 0], 
            [4, 2/9*np.pi], 
            [4, 4/9*np.pi], 
            [4, 6/9*np.pi], 
            [4, 8/9*np.pi], 
            [4, 10/9*np.pi], 
            [4, 12/9*np.pi], 
            [4, 14/9*np.pi], 
            [4, 16/9*np.pi], 
            [4, 2*np.pi]
            ])
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0],
            [1.3944673746950647, 2/9*np.pi],
            [1.4532809822903292, 4/9*np.pi],
            [1.520712695226306, 6/9*np.pi],
            [0.8129871265307131, 8/9*np.pi],
            [0.8129871265307131, 10/9*np.pi],
            [1.520712695226306, 12/9*np.pi],
            [1.4532809822903292, 14/9*np.pi],
            [1.3944673746950647, 16/9*np.pi],
            [1.3714590218125726, 2*np.pi]
            ])
        np.testing.assert_array_almost_equal(expected_hole, calculated_hole)
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)

    def test_no_slit_n5(self):
        '''
        Check some mirror limits with n=5 and with no slit inserted 
        - catch the main axes where x=0 or y=0
        '''
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 5), holein=True, slit=None, slit_center=0, orientation=0
            )
        expected_hole = np.array([
            [4, 0], 
            [4, np.pi/2], 
            [4, np.pi], 
            [4, 3*np.pi/2], 
            [4, 2*np.pi], 
            ])
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0],
            [1.4706289056333368, np.pi/2],
            [0.7512319991266359, np.pi],
            [1.4706289056333368, 3*np.pi/2],
            [1.3714590218125726, 2*np.pi]
            ])
        np.testing.assert_array_almost_equal(expected_hole, calculated_hole)
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)

    def test_3mm_slit(self):
        '''
        A mirror with a 3 mm slit inserted. Check the corners and some points on each edge.
        '''
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0], # +x axis
            [0.7512319991266359, np.pi], # -x axis
            [0.6511374590101888, np.pi/4], # 45 degrees +x, +y
            [0.3324469517442621, 3*np.pi/4], # 135 degrees +x, +y
            [0.3324469517442621, 5*np.pi/4], # 225 degrees +x, +y
            [0.3046926540153975, np.pi/2], # +y axis
            [0.740264374552439, 2.992952916455635], #slit corner -0.01
            [0.7603183610079051, 3.002952916455635], #slit corner (x,y)=(-10.75, 1.5)
            [0.759041372600021, 3.012952916455635], # slit corner +0.01
            [0.7603183610079051, 3.2802323907239512], #slit corner (x,y)=(-10.75, -1.5)
            [0.759041372600021, 3.2702323907239512], #slit corner -0.01
            [0.740264374552439, 3.2902323907239512], #slit corner +0.01
            [0.3046926540153975, 3*np.pi/2], # -y axis
            [0.6511374590101888, 7*np.pi/4], # 315 degrees +x, -y
            [1.387961189801988, 0.5880026035475675], # slit corner (y,z)=(1.5, 0.5)
            [1.3874190401273974, 0.5780026035475675], # slit corner -0.01 (y,z)=(1.5, 0.5)
            [1.2604138938978349, 0.5980026035475675], # slit corner +0.01 (y,z)=(1.5, 0.5)
            [1.387961189801988, 5.695182703632018], # slit corner (y,z)=(-1.5, 0.5)
            [1.3874190401273974, 5.705182703632019], # slit corner +0.01 (y,z)=(-1.5, 0.5)
            [1.2604138938978349, 5.685182703632018], # slit corner -0.01 (y,z)=(-1.5, 0.5)
            ])
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=expected_mirror_edge[:, 1], holein=True, slit=3, slit_center=0, orientation=0
            )
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)

    def test_0p8mm_slit(self):
        '''
        A mirror with a 0.8 mm slit inserted. Check the corners and some points on each edge.
        '''
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0], # +x axis
            [1.3714590218125726, 2*np.pi], # +x axis
            [0.7512319991266359, np.pi], # -x axis
            [0.08008558003365901, np.pi/2], # +y axis
            [0.08008558003365901, 3*np.pi/2], # -y axis
            [1.3727309261351577, 0.1612553355979321], # slit corner (x, 0.4, 0.5)
            [1.0622647376869492, 0.1712553355979321], # slit corner +0.01 (x, 0.4, z)
            [1.372578332865518, 0.1512553355979321], # slit corner -0.01 (x, y, 0.5)
            [0.7518784782278566, 3.1044005095049334], # slit corner (-10.75, 0.4 ,z)
            [0.7515774301358702, 3.1144005095049334], # slit corner +0.01 (-10.75, y ,z)
            [0.6809058866539073, 3.0944005095049334], # slit corner -0.01 (x, 0.4 ,z)
            [0.7518784782278566, 3.178784797674653], # slit corner (-10.75, -0.4 ,z)
            [0.6809058866539073, 3.188784797674653], # slit corner +0.01 (x, -0.4 ,z)
            [0.7515774301358702, 3.168784797674653], # slit corner -0.01 (-10.75, y ,z)
            [1.3727309261351577, 6.121929971581654], # slit corner (x, -0.4, 0.5)
            [1.372578332865518, 6.131929971581654], # slit corner +0.01 (x, y, 0.5)
            [1.0622647376869492, 6.111929971581654], # slit corner -0.01 (x, -0.4, z)
            ])
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=expected_mirror_edge[:, 1], holein=True, slit=0.8, slit_center=0, orientation=0
            )
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)
    
    def test_off_centre_3mm_slit(self):
        '''
        A mirror with an off-centre 3 mm slit inserted. Check the corners and some points on each edge.
        '''
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0], # +x axis
            [0.7512319991266359, np.pi], # -x axis
            [0.411516846067488, np.pi/2], # +y axis
            [0.20135792079033082, 3*np.pi/2], # -y axis
            [1.3990125208448267, 0.7669953331631361], # slit corner (x, 2, 0.5)
            [1.2931324309226961, 0.7769953331631361], # slit corner +0.01 (x, 2, z)
            [1.3983307321248761, 0.7569953331631361], # slit corner -0.01 (x, y, 0.5)
            [0.7673796503892008, 2.9576491970526178], # slit corner (-10.75, 2 ,z)
            [0.7656379287223979, 2.9676491970526178], # slit corner +0.01 (-10.75, y ,z)
            [0.7525347363355261, 2.9476491970526178], # slit corner -0.01 (x, 2 ,z)
            [0.7552715982942766, 3.2343489737779367], # slit corner (-10.75, -1 ,z)
            [0.7252259353121417, 3.2443489737779367], # slit corner +0.01 (x, -1 ,z)
            [0.7544439319383743, 3.2243489737779367], # slit corner -0.01 (-10.75, y ,z)
            [1.3791491318194802, 5.884662861513166], # slit corner (x, -1, 0.5)
            [1.378772446265661, 5.894662861513166], # slit corner +0.01 (x, y, 0.5)
            [1.2106314002043548, 5.874662861513166], # slit corner -0.01 (x, -1, z)
            ])
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=expected_mirror_edge[:, 1], holein=True, slit=3, slit_center=0.5, orientation=0
            )
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)
    
    def test_off_centre_2p5mm_slit(self):
        '''
        A mirror with an off-centre 2.5 mm slit inserted. Check the corners and some points on each edge. Off-centre more towards -y
        '''
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0], # +x axis
            [0.7512319991266359, np.pi], # -x axis
            [0.15056827277668602, np.pi/2], # +y axis
            [0.3575711036455103, 3*np.pi/2], # -y axis
            [1.3758594573582716, 0.3006763914001276], # slit corner (x, 0.75, 0.5)
            [1.170389926772985, 0.3106763914001276], # slit corner +0.01 (x, 0.75, z)
            [1.3755734771006127, 0.2906763914001276], # slit corner -0.01 (x, y, 0.5)
            [0.7535045315198744, 3.071938079861225], # slit corner (-10.75, 0.75 ,z)
            [0.7528974833480605, 3.081938079861225], # slit corner +0.01 (-10.75, y ,z)
            [0.7138530239410341, 3.061938079861225], # slit corner -0.01 (x, 0.75 ,z)
            [0.7635973537167893, 3.3029677640104844], # slit corner (-10.75, -1.75 ,z)
            [0.7464956862960416, 3.3129677640104844], # slit corner +0.01 (x, -1.75 ,z)
            [0.7620902804694006, 3.2929677640104844], # slit corner -0.01 (-10.75, y ,z)
            [1.393264797106423, 5.604241237343148], # slit corner (x, -1.75, 0.5)
            [1.3926493701581706, 5.614241237343148], # slit corner +0.01 (x, y, 0.5)
            [1.2780609545496133, 5.594241237343148], # slit corner -0.01 (x, -1.75, z)
            ])
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=expected_mirror_edge[:, 1], holein=True, slit=2.5, slit_center=-0.5, orientation=0
            )
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)
    
    def test_no_slit_rot90_clockwise(self):
        '''
        Check some mirror limits with n=10 and with no slit inserted. Mirror is rotated by 90 degrees clockwise
        '''
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 10), holein=True, slit=None, slit_center=0, orientation=np.pi/2
            )
        expected_hole = np.array([
            [4, 0], 
            [4, 2/9*np.pi], 
            [4, 4/9*np.pi], 
            [4, 6/9*np.pi], 
            [4, 8/9*np.pi], 
            [4, 10/9*np.pi], 
            [4, 12/9*np.pi], 
            [4, 14/9*np.pi], 
            [4, 16/9*np.pi], 
            [4, 2*np.pi]
            ]) + np.array([0, np.pi/2])
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0],
            [1.3944673746950647, 2/9*np.pi],
            [1.4532809822903292, 4/9*np.pi],
            [1.520712695226306, 6/9*np.pi],
            [0.8129871265307131, 8/9*np.pi],
            [0.8129871265307131, 10/9*np.pi],
            [1.520712695226306, 12/9*np.pi],
            [1.4532809822903292, 14/9*np.pi],
            [1.3944673746950647, 16/9*np.pi],
            [1.3714590218125726, 2*np.pi]
            ]) + np.array([0, np.pi/2])
        np.testing.assert_array_almost_equal(expected_hole, calculated_hole)
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)
    
    def test_no_slit_rot90_counterclockwise(self):
        '''
        Check some mirror limits with n=10 and with no slit inserted. Mirror is rotated by 90 degrees clockwise
        '''
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=np.linspace(0, 2*np.pi, 10), holein=True, slit=None, slit_center=0, orientation=-np.pi/2
            )
        expected_hole = np.array([
            [4, 0], 
            [4, 2/9*np.pi], 
            [4, 4/9*np.pi], 
            [4, 6/9*np.pi], 
            [4, 8/9*np.pi], 
            [4, 10/9*np.pi], 
            [4, 12/9*np.pi], 
            [4, 14/9*np.pi], 
            [4, 16/9*np.pi], 
            [4, 2*np.pi]
            ]) - np.array([0, np.pi/2])
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0],
            [1.3944673746950647, 2/9*np.pi],
            [1.4532809822903292, 4/9*np.pi],
            [1.520712695226306, 6/9*np.pi],
            [0.8129871265307131, 8/9*np.pi],
            [0.8129871265307131, 10/9*np.pi],
            [1.520712695226306, 12/9*np.pi],
            [1.4532809822903292, 14/9*np.pi],
            [1.3944673746950647, 16/9*np.pi],
            [1.3714590218125726, 2*np.pi]
            ]) - np.array([0, np.pi/2])
        np.testing.assert_array_almost_equal(expected_hole, calculated_hole)
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)

    def test_3mm_slit_rot45_CW(self):
        '''
        Check mirror limits with a 3 mm slit, rotated by 45 degrees in the clockwise direction
        '''
        expected_hole = np.array([
            [4, 0], 
            [4, np.pi], 
            [4, np.pi/4], 
            [4, 3*np.pi/4], 
            [4, 5*np.pi/4], 
            [4, np.pi/2], 
            [4, 2.992952916455635], 
            [4, 3.002952916455635], 
            [4, 3.012952916455635], 
            [4, 3.2802323907239512],
            [4, 3.2702323907239512],
            [4, 3.2902323907239512],
            [4, 3*np.pi/2],
            [4, 7*np.pi/4],
            [4, 0.5880026035475675],
            [4, 0.5780026035475675],
            [4, 0.5980026035475675],
            [4, 5.695182703632018],
            [4, 5.705182703632019],
            [4, 5.685182703632018],
            ]) + np.array([0, np.pi/4])
        expected_mirror_edge = np.array([
            [1.3714590218125726, 0], # +x axis
            [0.7512319991266359, np.pi], # -x axis
            [0.6511374590101888, np.pi/4], # 45 degrees +x, +y
            [0.3324469517442621, 3*np.pi/4], # 135 degrees +x, +y
            [0.3324469517442621, 5*np.pi/4], # 225 degrees +x, +y
            [0.3046926540153975, np.pi/2], # +y axis
            [0.740264374552439, 2.992952916455635], #slit corner -0.01
            [0.7603183610079051, 3.002952916455635], #slit corner (x,y)=(-10.75, 1.5)
            [0.759041372600021, 3.012952916455635], # slit corner +0.01
            [0.7603183610079051, 3.2802323907239512], #slit corner (x,y)=(-10.75, -1.5)
            [0.759041372600021, 3.2702323907239512], #slit corner -0.01
            [0.740264374552439, 3.2902323907239512], #slit corner +0.01
            [0.3046926540153975, 3*np.pi/2], # -y axis
            [0.6511374590101888, 7*np.pi/4], # 315 degrees +x, -y
            [1.387961189801988, 0.5880026035475675], # slit corner (y,z)=(1.5, 0.5)
            [1.3874190401273974, 0.5780026035475675], # slit corner -0.01 (y,z)=(1.5, 0.5)
            [1.2604138938978349, 0.5980026035475675], # slit corner +0.01 (y,z)=(1.5, 0.5)
            [1.387961189801988, 5.695182703632018], # slit corner (y,z)=(-1.5, 0.5)
            [1.3874190401273974, 5.705182703632019], # slit corner +0.01 (y,z)=(-1.5, 0.5)
            [1.2604138938978349, 5.685182703632018], # slit corner -0.01 (y,z)=(-1.5, 0.5)
            ])
        calculated_mirror_edge, calculated_hole = miop.mirror_outline(
            phi=expected_mirror_edge[:, 1], holein=True, slit=3, slit_center=0, orientation=np.pi/4
            )
        expected_mirror_edge += np.array([0, np.pi/4])
        np.testing.assert_array_almost_equal(expected_hole, calculated_hole)
        np.testing.assert_array_almost_equal(expected_mirror_edge, calculated_mirror_edge)


class ARMaskCalcTest(parameterized.TestCase):
    '''
    Test angle-resolved mirror masking function:
        centre of slit-done
        edges of 3 mm slit-done
        edges of different size slit-done
        edges of different size, rotated slit-done
        edges of off-centre slit -done
        edges of rotated slit-done
        edges of rotate, off-centre slit
        with and without hole
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
        mask_calc = miop.ar_mask_calc(
            theta, phi, holein=holein, slit=slit, slit_center=slit_center, 
            orientation=orientation)
        np.testing.assert_array_equal(expected, mask_calc)

    def test_edge_of_hole(self):
        phi = np.deg2rad(np.array([5, 5, 5, 40]))
        theta = np.deg2rad(np.array([3.9, 4, 4.1, 4.1]))
        mask = np.array([True, True, False, False])
        mask_calc = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            orientation=0
            )
        np.testing.assert_array_equal(mask, mask_calc)


    def test_edges_of_centred_3mm_slit(self):
        '''
        Centred 3 mm slit
        Hole in
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
        calculated = miop.ar_mask_calc(
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
        Hole in
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0.5,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_centred_2p5mm_slit(self):
        '''
        edges of different size slit: 2.5 mm
        Hole in
        '''
        xyz = np.array([
                        [0, 1.24, 4.843800161030593], # inside slit, +y
                        [0, 1.26, 4.838636171484688], # outside slit, +y
                        [0, -1.24, 4.843800161030593], # inside slit, -y
                        [0, -1.26, 4.838636171484688], # outside slit, -y
                         [-1, 1.24, 5.784669394183215], # inside slit
                         [-1, 1.26, 5.780346010404568], # outside slit
                         [-1, -1.24, 5.784669394183215], # inside slit
                         [-1, -1.26, 5.780346010404568], # outside slit
                         [1, 1.24, 3.6691143345499606], # inside slit
                         [1, 1.26, 3.6622943628277613], # outside slit
                         [1, -1.24, 3.6691143345499606], # inside slit
                         [1, -1.26, 3.6622943628277613], # outside slit
                         [-10.74, 1.24, 11.439510479037116], # top edge, +y slit edge inside mirror
                         [-10.74, 1.26, 11.437324862047069], # top edge, +y slit edge outside mirror
                         [-10.76, 1.24, 11.44824877437593], # top edge, +y slit edge outside mirror
                         [-10.76, 1.26, 11.44606482595656], # top edge, +y slit edge outside mirror
                         [-10.74, -1.24, 11.439510479037116], # top edge, -y slit edge inside mirror
                         [-10.74, -1.26, 11.437324862047069], # top edge, -y slit edge outside mirror
                         [-10.76, -1.24, 11.44824877437593], # top edge, -y slit edge outside mirror
                         [-10.76, -1.26, 11.44606482595656], # top edge, -y slit edge outside mirror
                         [2.32223, 1.24, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.32023, 1.24, 0.51], # inside mirror bottom +y slit edge inside mirror
                         [2.31723, 1.26, 0.49], # inside mirror bottom +y slit edge outside mirror
                         [2.31523, 1.26, 0.51], # inside mirror bottom +y slit edge outside mirror
                         [2.32223, -1.24, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.32023, -1.24, 0.51], # inside mirror bottom -y slit edge inside mirror
                         [2.31723, -1.26, 0.49], # inside mirror bottom -y slit edge outside mirror
                         [2.31523, -1.26, 0.51], # inside mirror bottom -y slit edge outside mirror
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=2.5,
            slit_center=0,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_centred_3mm_slit_hole_out(self):
        '''
        centred 3 mm slit
        Hole out
        '''
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=False,
            slit=3,
            slit_center=0,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_centred_3mm_slit_rotated(self):
        '''
        edges of rotated slit
        Hole in
        '''
        xyz = np.array([
                [1.49, 0, 4.77282934955776], # inside slit, +y
                [1.51, 0, 4.766539625346673], # outside slit, +y
                [-1.49, 0, 4.77282934955776], # inside slit, -y
                [-1.51, 0, 4.766539625346673], # outside slit, -y
                [1.49, 1, 5.725373350271578], # inside slit
                [1.51, 1, 5.720131117378342], # outside slit
                [-1.49, 1, 5.725373350271578], # inside slit
                [-1.51, 1, 5.720131117378342], # outside slit
                [1.49, -1, 3.574898599960564], # inside slit
                [1.51, -1, 3.5664968806939954], # outside slit
                [-1.49, -1, 3.574898599960564], # inside slit
                [-1.51, -1, 3.5664968806939954], # outside slit
                [0, -2.465, 0.5916079783099628], # inside mirror bottom
                [0, -2.485, 0.3872983346207433], # outside mirror bottom
                [0, -2.47399, 0.51], # inside mirror bottom
                [0, -2.47599, 0.49], # outside mirror bottom
                [0, 2.465, 0.5916079783099628], # wrong side of mirror
                [0, 2.485, 0.3872983346207433], # wrong side of mirror
                [0, 10.74, 11.50651989091402], # inside top edge mirror
                [0, 10.76, 11.515207336387824], # outside top edge mirror
                [1.49, 10.74, 11.409640660423973], # top edge, +y slit edge inside mirror
                [1.51, 10.74, 11.407011002011], # top edge, +y slit edge outside mirror
                [1.49, 10.76, 11.418401814614862], # top edge, +y slit edge outside mirror
                [1.51, 10.76, 11.415774174360667], # top edge, +y slit edge outside mirror
                [-1.49, 10.74, 11.409640660423973], # top edge, -y slit edge inside mirror
                [-1.51, 10.74, 11.407011002011], # top edge, -y slit edge outside mirror
                [-1.49, 10.76, 11.418401814614862], # top edge, -y slit edge outside mirror
                [-1.51, 10.76, 11.415774174360667], # top edge, -y slit edge outside mirror
                [1.49, -2.25398, 0.49], # inside mirror bottom +y slit edge outside mirror
                [1.49, -2.25198, 0.51], # inside mirror bottom +y slit edge inside mirror
                [1.51, -2.24798, 0.49], # inside mirror bottom +y slit edge outside mirror
                [1.51, -2.24598, 0.51], # inside mirror bottom +y slit edge outside mirror
                [-1.49, -2.25398, 0.49], # inside mirror bottom -y slit edge outside mirror
                [-1.49, -2.25198, 0.51], # inside mirror bottom -y slit edge inside mirror
                [-1.51, -2.24798, 0.49], # inside mirror bottom -y slit edge outside mirror
                [-1.51, -2.24598, 0.51], # inside mirror bottom -y slit edge outside mirror
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0,
            orientation=np.pi/2,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_centred_2p5mm_slit_rotated(self):
        '''
        edges of different size slit, rotated by pi/3
        '''
        xyz = np.array([
                        [1.0738715006927038, 0.62, 4.843800161030593], # inside slit, +y
                        [1.0911920087683924, 0.63, 4.838636171484688], # outside slit, +y
                        [-1.0738715006927038, -0.62, 4.843800161030593], # inside slit, -y
                        [-1.0911920087683924, -0.63, 4.838636171484688], # outside slit, -y
                        [0.5738715006927039, 1.486025403784439, 5.784669394183215], # inside slit
                        [0.5911920087683931, 1.4960254037844383, 5.780346010404568], # outside slit
                        [-1.5738715006927037, 0.2460254037844378, 5.784669394183215], # inside slit
                        [-1.5911920087683928, 0.23602540378443795, 5.780346010404568], # outside slit
                        [1.5738715006927038, -0.2460254037844385, 3.6691143345499606], # inside slit
                        [1.5911920087683926, -0.23602540378443845, 3.6622943628277613], # outside slit
                        [-0.5738715006927037, -1.4860254037844385, 3.6691143345499606], # inside slit
                        [-0.5911920087683927, -1.4960254037844383, 3.6622943628277613], # outside slit
                        [-4.296128499307294, 9.92111283664487, 11.439510479037116], # top edge, +y slit edge inside mirror
                        [-4.278807991231607, 9.93111283664487, 11.437324862047069], # top edge, +y slit edge outside mirror
                        [-4.306128499307295, 9.93843334472056, 11.44824877437593], # top edge, +y slit edge outside mirror
                        [-4.288807991231608, 9.948433344720558, 11.44606482595656], # top edge, +y slit edge outside mirror
                        [-6.443871500692703, 8.68111283664487, 11.439510479037116], # top edge, -y slit edge inside mirror
                        [-6.461192008768393, 8.671112836644868, 11.437324862047069], # top edge, -y slit edge outside mirror
                        [-6.453871500692706, 8.698433344720558, 11.44824877437593], # top edge, -y slit edge outside mirror
                        [-6.471192008768389, 8.688433344720561, 11.44606482595656], # top edge, -y slit edge outside mirror
                        [2.2349865006927034, -1.3911101734303366, 0.49], # inside mirror bottom +y slit edge outside mirror
                        [2.233986500692704, -1.3893781226227677, 0.51], # inside mirror bottom +y slit edge inside mirror
                        [2.249807008768393, -1.3767800464114146, 0.49], # inside mirror bottom +y slit edge outside mirror
                        [2.248807008768393, -1.3750479956038457, 0.51], # inside mirror bottom +y slit edge outside mirror
                        [0.08724349930729697, -2.6311101734303364, 0.49], # inside mirror bottom -y slit edge outside mirror
                        [0.08624349930729534, -2.629378122622768, 0.51], # inside mirror bottom -y slit edge inside mirror
                        [0.0674229912316071, -2.636780046411415, 0.49], # inside mirror bottom -y slit edge outside mirror
                        [0.06642299123160796, -2.6350479956038457, 0.51], # inside mirror bottom -y slit edge outside mirror
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=2.5,
            slit_center=0,
            orientation=np.pi/3,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_edges_of_offcentre_3mm_rotated_slit(self):
        '''
        edges of off-centre slit, moved 0.5 mm to the positive y and rotated by -pi/4
        Hole in
        '''
        # z = sqrt(10*(2.5-x)-y^2)
        xyz = np.array([
                    [-1.4071424945612299, 1.40714249456123, 4.586927075940928], # inside slit, +y
                    [-1.4212846301849602, 1.4212846301849604, 4.578198335590105], # outside slit, +y
                    [0.7000357133746818, -0.7000357133746822, 4.901010099969189], # inside slit, -y
                    [0.7141778489984129, -0.7141778489984132, 4.896927608204965], # outside slit, -y
                    [-2.1142492757477767, 0.7000357133746835, 5.571346336389436], # inside slit
                    [-2.1283914113715077, 0.7141778489984124, 5.564162111225732], # outside slit
                    [-0.007071067811865587, -1.4071424945612292, 5.832658056152443], # inside slit
                    [0.007071067811865346, -1.4212846301849609, 5.82922807925715], # outside slit
                    [-0.7000357133746821, 2.114249275747777, 3.3226344969015176], # inside slit
                    [-0.7141778489984124, 2.1283914113715072, 3.310573968362586], # outside slit
                    [1.4071424945612296, 0.007071067811864986, 3.7443156918187332], # inside slit
                    [1.4212846301849604, -0.007071067811865952, 3.738970446526691], # outside slit
                    [-9.00146932450475, -6.187184335382288, 11.333132841363856], # top edge, +y slit edge inside mirror
                    [-9.015611460128483, -6.173042199758558, 11.329602817398323], # top edge, +y slit edge outside mirror
                    [-9.015611460128481, -6.201326471006022, 11.341953094595304], # top edge, +y slit edge outside mirror
                    [-9.029753595752213, -6.187184335382289, 11.338425816664323], # top edge, +y slit edge outside mirror
                    [-6.894291116568843, -8.294362543318195, 11.46385188320226], # top edge, -y slit edge inside mirror
                    [-6.880148980945104, -8.308504678941935, 11.462107136124667], # top edge, -y slit edge outside mirror
                    [-6.908433252192571, -8.308504678941933, 11.472571638477573], # top edge, -y slit edge outside mirror
                    [-6.894291116568844, -8.32264681456566, 11.47082821770076], # top edge, -y slit edge outside mirror
                    [0.06362546817116511, 2.877910457293625, 0.49], # inside mirror bottom +y slit edge outside mirror
                    [0.062211254608792475, 2.876496243731251, 0.51], # inside mirror bottom +y slit edge inside mirror
                    [0.04382647829794253, 2.8863957386678627, 0.49], # inside mirror bottom +y slit edge outside mirror
                    [0.04241226473556947, 2.8849815251054904, 0.51], # inside mirror bottom +y slit edge outside mirror
                    [2.3815214969006684, 0.9814500701513039, 0.49], # inside mirror bottom -y slit edge outside mirror
                    [2.3801072833382957, 0.9800358565889307, 0.51], # inside mirror bottom -y slit edge inside mirror
                    [2.392835205399654, 0.9644795074028256, 0.49], # inside mirror bottom -y slit edge outside mirror
                    [2.3914209918372804, 0.9630652938404543, 0.51], # inside mirror bottom -y slit edge outside mirror
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
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0.5,
            orientation=-np.pi/4,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_near_slit(self):
        '''
        centre of slit
        edges of 3 mm slit
        edges of different size slit
        edges of off-centre slit
        edges of rotated slit
        edges of rotate, off-centre slit
        with and without hole
        '''
        theta = np.array([
                        np.pi/4, #45 degrees on +x
                        np.pi/4, #45 degrees on -x
                        np.pi/4, #45 degrees on +y
                        np.pi/4, #45 degrees on -y
                        ])
        phi = np.array([
                    0,
                    np.pi,
                    np.pi/2,
                    -np.pi/2,
                    ])
        expected = np.array([False, True, True, True])
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=3,
            slit_center=0,
            orientation=0,
            )
        np.testing.assert_array_equal(expected, calculated)

    def test_single_value(self):
        '''
        Feed in a single value to the mirror mask calculation 
        - should produce valid output
        '''
        theta = np.pi/6
        phi = -2
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=0,
            orientation=0,
            )
        expected=False
        np.testing.assert_array_equal(expected, calculated)
        
        
    def test_n_by_n_array(self):
        '''
        N by M array into the mask calculation
        '''
        theta_phi = np.array([
            [np.pi/4, 0],
            [np.pi/4, 0.1],
            [np.pi/4, -0.1],
            [np.pi/2, 0.1], #T
            [np.pi/3, 0.1],
            [np.pi/3, np.pi],
            [np.pi/3, -np.pi],
            [np.pi/3, np.pi/2],
            [np.pi/3, -np.pi/2],
            [0, np.pi/2], #T
            [0, np.pi/6], #T
            [0.1, np.pi/6],
            [np.pi/6, 3*np.pi/2],
            [np.pi/6, 1],
            [np.pi/6, 2],
            [np.pi/6, -1],
            [np.pi/6, -2],
            [np.pi/2, 1], #T
            [3, 1],  # T
            [3, -1],  # T
            ])
        theta = np.reshape(theta_phi[:, 0], (4, 5))
        phi = np.reshape(theta_phi[:, 1], (4, 5))
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=0,
            orientation=0,
            )
        expected = np.reshape(np.array([
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            ]), (4,5))
        np.testing.assert_array_equal(calculated, expected)
        
    def test_n_by_1_array(self):
        '''
        N by 1 array into the mask calculation. Should produce a valid result.
        '''
        theta_phi = np.array([
            [np.pi/4, 0],
            [np.pi/4, 0.1],
            [np.pi/4, -0.1],
            [np.pi/2, 0.1], #T
            [np.pi/3, 0.1],
            [np.pi/3, np.pi],
            [np.pi/3, -np.pi],
            [np.pi/3, np.pi/2],
            [np.pi/3, -np.pi/2],
            [0, np.pi/2], #T
            [0, np.pi/6], #T
            [0.1, np.pi/6],
            [np.pi/6, 3*np.pi/2],
            [np.pi/6, 1],
            [np.pi/6, 2],
            [np.pi/6, -1],
            [np.pi/6, -2],
            [np.pi/2, 1], #T
            [3, 1],  # T
            [3, -1],  # T
            ])
        theta = np.reshape(theta_phi[:, 0], (20, 1))
        phi = np.reshape(theta_phi[:, 1], (20, 1))
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=0,
            orientation=0,
            )
        expected = np.reshape(np.array([
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            ]), (20, 1))
        np.testing.assert_array_equal(calculated, expected)
    
    def test_1_by_n_array(self):
        '''
        1 by N array into the mask calculation. Should produce a valid result.
        '''
        theta_phi = np.array([
            [np.pi/4, 0],
            [np.pi/4, 0.1],
            [np.pi/4, -0.1],
            [np.pi/2, 0.1], #T
            [np.pi/3, 0.1],
            [np.pi/3, np.pi],
            [np.pi/3, -np.pi],
            [np.pi/3, np.pi/2],
            [np.pi/3, -np.pi/2],
            [0, np.pi/2], #T
            [0, np.pi/6], #T
            [0.1, np.pi/6],
            [np.pi/6, 3*np.pi/2],
            [np.pi/6, 1],
            [np.pi/6, 2],
            [np.pi/6, -1],
            [np.pi/6, -2],
            [np.pi/2, 1], #T
            [3, 1],  # T
            [3, -1],  # T
            ])
        theta = np.reshape(theta_phi[:, 0], (1, 20))
        phi = np.reshape(theta_phi[:, 1], (1, 20))
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=0,
            orientation=0,
            )
        expected = np.reshape(np.array([
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            ]), (1, 20))
        np.testing.assert_array_equal(calculated, expected)
    
    def test_3d_array(self):
        '''
        N by M by P array into the mask calculation. Should produce a valid result.
        '''
        theta_phi = np.array([
            [np.pi/4, 0],
            [np.pi/4, 0.1],
            [np.pi/4, -0.1],
            [np.pi/2, 0.1], #T
            [np.pi/3, 0.1],
            [np.pi/3, np.pi],
            [np.pi/3, -np.pi],
            [np.pi/3, np.pi/2],
            [np.pi/3, -np.pi/2],
            [0, np.pi/2], #T
            [0, np.pi/6], #T
            [0.1, np.pi/6],
            [np.pi/6, 3*np.pi/2],
            [np.pi/6, 1],
            [np.pi/6, 2],
            [np.pi/6, -1],
            [np.pi/6, -2],
            [np.pi/2, 1], #T
            [3, 1],  # T
            [3, -1],  # T
            ])
        theta = np.reshape(theta_phi[:, 0], (2, 2, 5))
        phi = np.reshape(theta_phi[:, 1], (2, 2, 5))
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=0,
            orientation=0,
            )
        expected = np.reshape(np.array([
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            ]), (2, 2, 5))
        np.testing.assert_array_equal(calculated, expected)


class ParabolaPositionTest(parameterized.TestCase):
    '''
    Test the parabola_position function in mirror_operations
    '''
    @parameterized.named_parameters(
        ('positive X axis', np.array([[1, 0, 0]]), np.array([[2.5, 0, 0]])),
        ('positive Y axis', np.array([[0, 1, 0]]), np.array([[0, 5, 0]])),
        ('negative Y axis', np.array([[0, -1, 0]]), np.array([[0, -5, 0]])),
        ('positive Z axis', np.array([[0, 0, 1]]), np.array([[0, 0, 5]])),
        ('negative Z axis', np.array([[0, 0, -1]]), np.array([[0, 0, -5]])),
        ('off axis in-out equal', np.array([[2.3, -1, -1]]), np.array([[2.3, -1, -1]])),
        ('off axis in-out not equal', np.array([[-1, 5, 3]]), np.array([[-1.017070556338178, 5.085352781690894, 3.051211669014535]])),
        ('2x3 input', np.array([[0, 1, 0], [0, 0, 1]]), np.array([[0, 5, 0], [0, 0, 5]])),
        ('from Matlab (1)', np.array([[2.480183375807916, -0.052301218492773, 0.442075586823127]]), np.array([[2.480183375807916, -0.052301218492773, 0.442075586823127]])),
        ('ones', np.array([[1, 1, 1]]), np.array([[1.830127018922193, 1.830127018922193, 1.830127018922193]])),
#        ('three-vector input', np.array([-1, 0, 0]), np.array([-2.5, 0, 0]))
#        ('2x2x3 input',
#         np.array([[[0, 1, 0], [0, 0, 1]], [[0, -1, 0], [0, 0, -1]]]),
#         np.array([[[0, 5, 0], [0, 0, 5]], [[0, -5, 0], [0, 0, -5]]]),
#         ),
    )
    def test_parabola_positions(self, direction, expected_position):
        '''
        check that parabola positions are calculated correctly
        '''
        position = miop.parabola_position(direction)
        np.testing.assert_allclose(expected_position, position, atol=1e-7)

    def test_parabola_position_negative_X(self):
        '''
        check the parabola positions returns NaN on negative x axis
        '''
        xyz = np.array([[-1, 0, 0]])
        position_expected = np.array([[np.nan, np.nan, np.nan]])
        position = miop.parabola_position(xyz);
        self.assertTrue(np.all(np.isnan(position)))


class ParabolaNormalsTest(parameterized.TestCase):
    '''
    Test the parabola_normals function in mirror_operations
    '''
    def test_negative_x_axis(self):
        '''
        Check that the parabola normal on the negative x-axis does not exist
        '''
        direction = np.array([[-1, 0, 0]])
        parabola_position = miop.parabola_position(direction)
        calculated_normal = miop.parabola_normals(parabola_position)
        self.assertTrue(np.all(np.isnan(calculated_normal)))

    @parameterized.named_parameters(
        dict(testcase_name='positive X axis',
             direction=np.array([[1, 0, 0]]),
             expected_normal=np.array([[-1, 0, 0]])
        ),
        dict(testcase_name='positive Y axis',
             direction=np.array([[0, 1 , 0]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[-1, -1/5*5, 0]])
        ),
        dict(testcase_name='negative Y axis',
             direction=np.array([[0, -1, 0]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[-1, 1/5*5, 0]])
        ), 
        dict(testcase_name='positive Z axis',
             direction=np.array([[0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[-1, 0, -1/5*5]])
        ),
        dict(testcase_name='negative Z axis',
             direction=np.array([[0, 0, -1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[-1, 0, 1/5*5]])
        ),
        dict(testcase_name='2 by 3 array input',
             direction=np.array([[0, 0, -1], [0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[-1, 0, 1/5*5], [-1, 0, -1/5*5]])
        ),
        dict(testcase_name='from Matlab',
            direction=np.array([[2.480183375807916, -0.052301218492773, 0.442075586823127]]),
            expected_normal=np.array([[-0.996060082509876, 0.010419031201456, -0.088066769097329]])
        ),
        dict(testcase_name='incident ones',
            direction=np.array([[1, 1, 1]]),
            expected_normal=np.array([[-0.8880738338665906, -0.3250575838228477, -0.3250575838228477]])
        )
    )
    def test_parabola_normals_directions(self, direction, expected_normal):
        '''
        check that the parabola normals are calculated correctly in various
        directions
        '''
        position = miop.parabola_position(direction)
        normal = miop.parabola_normals(position)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-7)

    @parameterized.named_parameters(
        dict(testcase_name='positive X axis',
             position=np.array([[1, 0, 0]]),
             expected_normal=np.array([[-1, 0, 0]])
        ),
        dict(testcase_name='positive Y axis',
             position=np.array([[0, 1, 0]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[-1, -1/5, 0]])
        ),
        dict(testcase_name='negative Y axis',
             position=np.array([[0, -1, 0]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[-1, 1/5, 0]])
        ),
        dict(testcase_name='positive Z axis',
             position=np.array([[0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[-1, 0, -1/5]])
        ),
        dict(testcase_name='negative Z axis',
             position=np.array([[0, 0, -1]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[-1, 0, 1/5]])
        ),
        dict(testcase_name='positive Y, positive Z',
             position=np.array([[0, 1, 3]]),
             expected_normal=1/np.sqrt(1+10/25)*np.array([[-1, -1/5, -3/5]])
        ),
        dict(testcase_name='positive Y, negative Z',
             position=np.array([[0, 1, -3]]),
             expected_normal=1/np.sqrt(1+10/25)*np.array([[-1, -1/5, 3/5]])
        ),
        dict(testcase_name='negative Y, positive Z',
             position=np.array([[0, -5, 3]]),
             expected_normal=1/np.sqrt(1+34/25)*np.array([[-1, 1, -3/5]])
        ),
        dict(testcase_name='negative Y, negative Z',
             position=np.array([[0, -5, -3]]),
             expected_normal=1/np.sqrt(1+34/25)*np.array([[-1, 1, 3/5]])
        ),
        dict(testcase_name='from Matlab',
            position=np.array([[2.480183375807916, -0.052301218492773, 0.442075586823127]]),
            expected_normal=np.array([[-0.996060082509876, 0.010419031201456, -0.088066769097329]])
            )
    )
    def test_parabola_normals_positions(self, position, expected_normal):
        '''
        check that the parabola normals are calculated correctly given various
        (y,z) positions on the parameterized parabola
        '''
        normal = miop.parabola_normals(position)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-7)


class ParabolaSurfacePolarizationTest(parameterized.TestCase):
    '''
    Test the surface polarization direction for parabolas
    '''
    @parameterized.named_parameters(
#        dict(testcase_name='positive X axis',
#             theta=np.array([np.pi/2]),
#             phi=np.array([0]),
#             expected_p=np.array([[0, 0, 0]]),
#             expected_s=np.array([[0, 0, 0]]),
#        ),
        dict(testcase_name='positive Y axis',
             theta=np.array([np.pi/2]),
             phi=np.array([np.pi/2]),
             expected_p=np.array([[1, 0, 0]]),
             expected_s=np.array([[0, 0, 1]]),
        ),
#        dict(testcase_name='negative X axis',
#             theta=np.array([np.pi/2]),
#             phi=np.array([np.pi]),
#             expected_p=np.array([[0, 0, 0]]),
#             expected_s=np.array([[0, 0, 0]]),
#        ), 
        dict(testcase_name='negative Y axis', 
             theta=np.array([np.pi/2]),
             phi=np.array([3*np.pi/2]),
             expected_p=np.array([[1, 0, 0]]),
             expected_s=np.array([[0, 0, -1]]),
        ),
        dict(testcase_name='negative Y axis backwards',
             theta=np.array([np.pi/2]),
             phi=np.array([-np.pi/2]),
             expected_p=np.array([[1, 0, 0]]),
             expected_s=np.array([[0, 0, -1]]),
        ),
        dict(testcase_name='45 degrees off positive z',
             theta=np.array([np.pi/4]),
             phi=np.array([np.pi/2]),
             expected_p=np.array([[1, 0, 0]]),
             expected_s=np.array([[0, -1/np.sqrt(2), 1/np.sqrt(2)]]),
        ),
        dict(testcase_name='2x3 array',
             theta=np.array([np.pi/2, np.pi/4]),
             phi=np.array([-np.pi/2, np.pi/2]),
             expected_p=np.array([[1, 0, 0], [1, 0, 0]]),
             expected_s=np.array([[0, 0, -1], [0, -1/np.sqrt(2), 1/np.sqrt(2)]]),
         )
    )
    def test_surface_polarization_directions(self,
                                             theta,
                                             phi,
                                             expected_p,
                                             expected_s):
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        np.testing.assert_allclose(p, expected_p, atol=1e-7)
        np.testing.assert_allclose(s, expected_s, atol=1e-7)

    def test_output_dimensions_4x3(self):
        '''
        test the output dimensions are 4x3, given input of size 4
        '''
        theta = np.array([1, 2, 3, 4])
        phi = np.array([5, 6, 7, 8])
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        self.assertEqual(np.shape(p), (4,3))
        self.assertEqual(np.shape(s), (4,3))

    def test_output_dimensions_2x4x3(self):
        '''
        test the output dimensions are 2x4x3, given input of size 2x4
        '''
        theta = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        phi = np.array([[5, 6, 7, 8], [5, 6, 7, 8]])
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        self.assertEqual(np.shape(p), (2, 4, 3))
        self.assertEqual(np.shape(s), (2, 4, 3))

class GetMirrorReflectionCoefficientsTest(parameterized.TestCase):
    '''
    Test the calculation of reflection coefficients for an electric field
     impinging on the mirror
    '''
    @parameterized.named_parameters(
        dict(testcase_name='+x, -y',
             k_vector=np.array([1, -1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
        ),
        dict(testcase_name='-x, -y',
             k_vector=np.array([-1, -1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
        ),
        dict(testcase_name='+x, +y',
             k_vector=np.array([1, 1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
        ),
        dict(testcase_name='-x, +y',
             k_vector=np.array([-1, 1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
        ),
        dict(testcase_name='Brewster angle',
             k_vector=np.array([2, -1, 0]),
             normal=np.array([0, 1, 0]),
             expected_r_s=np.array([-0.6]),
             expected_r_p=np.array([0]),
        ),
        dict(testcase_name='not nice axis',
             k_vector=np.array([1, 1, 3]),
             normal=np.array([1, 2.5, 1]),
             expected_r_s=np.array([-0.46356686111079304]),
             expected_r_p=np.array([-0.18900224550388287]),
        ),
    )
    def test_incoming_vector_angles(self, k_vector, normal, expected_r_s, expected_r_p):
        '''
        Incoming vector impinges from different quadrants
        '''
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75, thetacutoffhole=4., dielectric=np.array([[0.5, 4, 0], [1.5, 4, 0], [2.5, 4, 0], [3.5, 4, 0], [4.5, 4, 0]]))
        wavelength = 800e-9
        n_environment = 1
        r_s, r_p = miop.get_mirror_reflection_coefficients(wavelength, normal, k_vector, mirror=mirror, n_environment=n_environment)
        np.testing.assert_allclose(r_s, expected_r_s, atol=1e-7)
        np.testing.assert_allclose(r_p, expected_r_p, atol=1e-7)

    def test_mirror_refractive_index_is_zero(self):
        '''
        the refractive index of the mirror is 0
        '''
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75, thetacutoffhole=4., dielectric=np.array([[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0], [3.5, 0, 0], [4.5, 0, 0]]))
        wavelength=800e-9
        n_environment = 1
        k_vector=np.array([1, -1, 0])/np.sqrt(2)
        normal=np.array([1, 0, 0])
        self.assertRaisesRegex(
            ValueError,
            "Mirror refractive index cannot be 0",
            miop.get_mirror_reflection_coefficients,
            wavelength,
            normal,
            k_vector,
            mirror,
            n_environment)


class MirrorRefractiveIndexTest(parameterized.TestCase):
    '''
    Testing the interpolation of the mirror refractive index
    - wavelength greater than that existing in the provided dielectric
    - wavelength less than the minimum in the provided dielectric
    - wavelength inside the provided dielectric
    '''
    @parameterized.named_parameters(
        dict(testcase_name='too high wavelength',
             wavelength=182,
            ),
        dict(testcase_name='too low wavelength',
             wavelength=1600,
            ),
        )
    def test_outside_wavelength_limits(self, wavelength):
        '''
        Testing wavelength values outside the range given by the default dielectric
        '''
        with self.assertRaises(ValueError):
            n = miop.get_mirror_refractive_index(wavelength)

    @parameterized.named_parameters(
        dict(testcase_name='1p55 eV',
             wavelength=799.89802*1e-9,
             expected_n = 2.7999999999999994 + 1j * 8.45000000000000
            ),
        dict(testcase_name='5p525 eV',
             wavelength=224.40578*1e-9,
             expected_n = 0.1533972647627147 + 1j * 2.6270656268891153
#             expected_dielectric = -6.877943087145617 + 1j * 0.8059693630338731
            ),
        dict(testcase_name='6p75 eV',
             wavelength=1.836802939751115e-07,
             expected_n = 0.101999952506306 + 1j * 2.07000096384308
            ),
        dict(testcase_name='0p5 eV',
             wavelength=2.4796839686640052e-06,
             expected_n = 3.07 + 1j * 25.6
            ),
        )
    def test_inside_wavelength_limits(self, wavelength, expected_n):
        '''
        Testing wavelength values inside the range given by the default dielectric
        '''
        calculated_n = miop.get_mirror_refractive_index(wavelength)
        self.assertAlmostEqual(expected_n, calculated_n, places=4)

    @parameterized.named_parameters(
        dict(testcase_name='2 eV',
             wavelength=619.92097*1e-9,
             expected_n = 0.895977476129838 + 1j * 1.6741492280355401
            ),
        dict(testcase_name='4 eV',
             wavelength=309.960496083001*1e-9,
             expected_n = 0.248098393402356 + 1j * 2.01532945515338
            ),
        )
    def test_different_dielectric(self, wavelength, expected_n):
        '''
        using a different dielectric function
        '''
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75,
            thetacutoffhole=4.,
            dielectric=np.array([[1, -1, 4],[2, -2, 3],[3,-3, 2],[4, -4, 1]])
            )
        calculated_n = miop.get_mirror_refractive_index(wavelength, mirror=mirror)
        self.assertAlmostEqual(expected_n, calculated_n, places=4)

    def test_interpolated_value_2p5eV(self):
        '''
        check a value interpolated between two points is within the range expected
        - not interested in testing the accuracy of the interpolation here
        '''
        wavelength=495.93677*1e-9
        n_high_limit = 0.895977476129838 + 1.67414922803554j
        n_low_limit = 0.550250522700337 + 1.81735402102397j
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75,
            thetacutoffhole=4.,
            dielectric=np.array([[1, -1, 4],[2, -2, 3],[3,-3, 2],[4, -4, 1]])
            )
        calculated_n = miop.get_mirror_refractive_index(wavelength, mirror=mirror)
        self.assertTrue(calculated_n < n_high_limit)
        self.assertTrue(calculated_n > n_low_limit)


class MirrorReflectedFieldTest(parameterized.TestCase):
    '''
    Testing the calculation of electric field vectors reflected off the mirror
    - output is the same shape as the input
    - output is correct given a normal input
    '''
    def testOutputShape4x3(self):
        '''
        Check the shape of the output field is the same as the shape of the input field, 4x3 input
        '''
        incident_direction = np.array([[1, 0, 0], [2, -3, 0], [-1, -1, -1], [2, 1, 0]])
        incident_e = np.array([[0, 1, 0], [1, 5, 2], [1, -1, 0.5], [5, -2, 1]])
        wavelength = 799.89802*1e-9
        n_environment = 1
        calculated_reflected_e_s, calculated_reflected_e_p, _ = miop.get_mirror_reflected_field(
            incident_direction,
            incident_e,
            wavelength,
            n_environment
            )
        expected_reflected_e_shape = np.shape(incident_direction)
        np.testing.assert_equal(np.shape(calculated_reflected_e_s), expected_reflected_e_shape)
        np.testing.assert_equal(np.shape(calculated_reflected_e_p), expected_reflected_e_shape)

    def tes
    tOutputShape1x3(self):
        '''
        Check the shape of the output field is the same as the shape of the input field, 1x3 input
        '''
        incident_direction = np.array([[1, 0, 0]])
        incident_e = np.array([[0, 1, 0]])
        wavelength = 799.89802*1e-9
        n_environment = 1
        calculated_reflected_e_s, calculated_reflected_e_p, _ = miop.get_mirror_reflected_field(
            incident_direction,
            incident_e,
            wavelength,
            n_environment
            )
        expected_reflected_e_shape = np.shape(incident_direction)
        np.testing.assert_equal(np.shape(calculated_reflected_e_s), expected_reflected_e_shape)
        np.testing.assert_equal(np.shape(calculated_reflected_e_p), expected_reflected_e_shape)

    def testSingleRealVector(self):
        '''
        Check the output is as expected given a real incident electric field 
        with values determined by a failed property test
        '''
        incident_direction = np.array([[1, 1, 1]])
        incident_e = np.array([[1, -1, 0]])
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        n_environment = 1
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        expected_e_s = (-0.461575 - 0.0882315j) * np.array([[0, -1, 1]])
        expected_e_p = np.array([[5.28768253e-10+1.29162216e-10j, -7.77563951e-01-1.89935539e-01j, -7.77563951e-01-1.89935539e-01j]])
#        expected_e_p = (-0.897854 - 0.219319j) * np.array([[1, -0.5, -0.5]])
        np.testing.assert_allclose(reflected_e_s, expected_e_s, atol=1e-5)
        np.testing.assert_allclose(reflected_e_p, expected_e_p, atol=1e-5)
#    @parameterized.named_parameters(
#        dict(testcase_name='Positive x-axis',
#             incident_direction=np.array([[1, 0, 0]]),
#             incident_e=np.array([[0, 1, 0]]),
#             wavelength=1800e-9,
#             expected_e=np.array([[]]),
#             )
#        def test_single_values(self, incident_direction, incident_e, wavelength, expected_e):
#            calculated_e = miop.get_mirror_reflected_field(incident_direction,
#                incident_e,
#                wavelength,
#                n_environment=1,
#                mirror=miop.AMOLF_MIRROR
#                )
#            np.testing.assert_allclose(calculated_e, expected_e)

class MuellerMatrixEllipsoidalTest(parameterized.TestCase):
    '''
    Testing the Mueller matrix calculation for an ellipsoidal mirror
    - all zeros
    - all real ones
    - all imaginary ones
    - mixed real + imaginary ones
    - arbitrary values
    '''
    @parameterized.named_parameters(
        dict(testcase_name='All zeros',
             e_h_h = 0,
             e_h_v = 0,
             e_v_h = 0,
             e_v_v = 0,
             expected_m = (
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0)
            ),
        dict(testcase_name='real ones',
             e_h_h = 1,
             e_h_v = 1,
             e_v_h = 1,
             e_v_v = 1,
             expected_m = (
                2, 0, 2, 0,
                0, 0, 0, 0,
                2, 0, 2, 0,
                0, 0, 0, 0
                )
            ),
        dict(testcase_name='Imaginary ones',
             e_h_h = 1j,
             e_h_v = 1j,
             e_v_h = 1j,
             e_v_v = 1j,
             expected_m = (
                2, 0, 2, 0,
                0, 0, 0, 0,
                2, 0, 2, 0,
                0, 0, 0, 0
                )
            ),
        dict(testcase_name='Mixed real and imag ones positive',
             e_h_h = 1+1j,
             e_h_v = 1+1j,
             e_v_h = 1+1j,
             e_v_v = 1+1j,
             expected_m = (
                4, 0, 4, 0,
                0, 0, 0, 0,
                4, 0, 4, 0,
                0, 0, 0, 0)
            ),
        dict(testcase_name='Mixed real and imag ones negative',
             e_h_h = -1-1j,
             e_h_v = -1-1j,
             e_v_h = -1-1j,
             e_v_v = -1-1j,
             expected_m = (
                4, 0, 4, 0,
                0, 0, 0, 0,
                4, 0, 4, 0,
                0, 0, 0, 0)
            ),
        dict(testcase_name='Mixed real and imag ones mixed case',
             e_h_h = -1+1j,
             e_h_v = +1-1j,
             e_v_h = +1+1j,
             e_v_v = -1-1j,
             expected_m = (
                4, 0, 0, 4,
                0, 0, 0, 0,
                -4, 0, 0, -4,
                0, 0, 0, 0)
            ),
        dict(testcase_name='arbitrary values',
             e_h_h = -5 + 3j,
             e_h_v = +0.6 + 0.7j,
             e_v_h = -2.1 - 0.8j,
             e_v_v = 1.1 + 0.1j,
             expected_m = (
                20.56, 14.29, 8.83, -9.59,
                18.49, 14.66, 7.37, -11.01,
                -3.29, 1.49, -7.02, 2.81,
                -4.63, -5.97, -4.79, -3.38)
            ),
        )
    def test_single_values(self, e_h_h, e_h_v, e_v_h, e_v_v, expected_m):
        '''
        Testing single value inputs
        '''
        calculated_m = miop.mueller_matrix_ellipsoidal(e_h_h, e_h_v, e_v_h, e_v_v)
        np.testing.assert_allclose(calculated_m, expected_m, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
