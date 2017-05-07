import numpy as np
from matplotlib import pyplot as plt
from sys import exit
def k_test01 ( x_num, x, t ):

  k_value = np.zeros ( x_num )

  for i in range ( 0, x_num ):
    k_value[i] = 1.0

  return k_value
def bc_test01 ( x_num, x, t, u ):
  k = 1.0
  u[0]       = np.exp(-x[0]**2 / (2+4*k*t)) / np.sqrt((2+4*k*t)*np.pi)
  u[x_num-1] = np.exp(-x[-1]**2 / (2+4*k*t)) / np.sqrt((2+4*k*t)*np.pi)

  return u
def bc_test02 ( x_num, x, t, u ):
  u[0]       = 0.0
  u[x_num-1] = 0.0

  return u
def rhs_test01 ( x_num, x, t ):

  rhs_value = np.zeros ( x_num )

  return rhs_value
  
def reference_to_physical ( element, element_node, node_x, reference_num, \
  reference_x ):

  physical_x = np.zeros ( reference_num )

  for i in range ( 0, reference_num ):
    a = node_x[element_node[0,element] ]
    b = node_x[element_node[1,element] ]

    physical_x[i] = ( ( 1.0 - reference_x[i]             ) * a   \
                    + (       reference_x[i] - ( - 1.0 ) ) * b ) \
                    / ( 1.0                  - ( - 1.0 ) )

  return physical_x
def quadrature_set ( quad_num ):

  if ( quad_num == 1 ):

    quad_x = np.array ( [ \
      0.0 ] )

    quad_w = np.array ( [ \
      2.0 ] )

  elif ( quad_num == 2 ):

    quad_x = np.array ( [ \
      - 0.577350269189625764509148780502, \
        0.577350269189625764509148780502 ] )

    quad_w = np.array ( [ \
      1.0, \
      1.0 ] )

  elif ( quad_num == 3 ):

    quad_x = np.array ( [ \
      - 0.774596669241483377035853079956, \
        0.0, \
        0.774596669241483377035853079956 ] )

    quad_w = np.array ( [ \
      5.0 / 9.0, \
      8.0 / 9.0, \
      5.0 / 9.0 ] )

  elif ( quad_num == 4 ):

    quad_x = np.array ( [ \
      - 0.861136311594052575223946488893, \
      - 0.339981043584856264802665759103, \
        0.339981043584856264802665759103, \
        0.861136311594052575223946488893 ] )

    quad_w = np.array ( [ \
      0.347854845137453857373063949222, \
      0.652145154862546142626936050778, \
      0.652145154862546142626936050778, \
      0.347854845137453857373063949222 ] )

  elif ( quad_num == 5 ):

    quad_x = np.array ( [ \
      - 0.906179845938663992797626878299, \
      - 0.538469310105683091036314420700, \
        0.0, \
        0.538469310105683091036314420700, \
        0.906179845938663992797626878299 ] )

    quad_w = np.array ( [ \
      0.236926885056189087514264040720, \
      0.478628670499366468041291514836, \
      0.568888888888888888888888888889, \
      0.478628670499366468041291514836, \
      0.236926885056189087514264040720 ] )

  elif ( quad_num == 6 ):
    quad_x = np.array ( [ \
      - 0.932469514203152027812301554494, \
      - 0.661209386466264513661399595020, \
      - 0.238619186083196908630501721681, \
        0.238619186083196908630501721681, \
        0.661209386466264513661399595020, \
        0.932469514203152027812301554494 ] )

    quad_w = np.array ( [ \
      0.171324492379170345040296142173, \
      0.360761573048138607569833513838, \
      0.467913934572691047389870343990, \
      0.467913934572691047389870343990, \
      0.360761573048138607569833513838, \
      0.171324492379170345040296142173 ] )

  else:

    print ( '' )
    print ( 'QUADRATURE_SET - Fatal error!' )
    print ( '  The requested order %d is not available.' % ( quad_num ) )
    exit("Fatal Error")
  return quad_w, quad_x
def basis_function ( index, element, node_x, point_x ):
  b    = 0.0
  dbdx = 0.0

  if ( index == element ):
    b    = ( node_x[element+1] - point_x ) / ( node_x[element+1] - node_x[element] )
    dbdx =                     - 1.0       / ( node_x[element+1] - node_x[element] )
  elif ( index == element + 1 ):
    b    = ( point_x - node_x[element] )   / ( node_x[element+1] - node_x[element] )
    dbdx = + 1.0                           / ( node_x[element+1] - node_x[element] )

  return b, dbdx

def assemble_mass ( node_num, node_x, element_num, element_node, quad_num ):

  c = np.zeros ( ( node_num, node_num ) )
#
#  Get the quadrature weights and nodes.
#
  reference_w, reference_q = quadrature_set ( quad_num )
#
#  Consider each ELEMENT.
#
  for element in range ( 0, element_num ):

    element_x = np.zeros ( 2 )
    element_x[0] = node_x[element_node[0,element]]
    element_x[1] = node_x[element_node[1,element]]

    element_q = reference_to_physical ( element, element_node, node_x, \
      quad_num, reference_q )

    element_area = element_x[1] - element_x[0]

    element_w = np.zeros ( quad_num )
    for quad in range ( 0, quad_num ):
      element_w[quad] = ( element_area / 2.0 ) * reference_w[quad]
#
#  Consider the QUAD-th quadrature point in the element.
#
    for quad in range ( 0, quad_num ):
#
#  Consider the TEST-th test function.
#
#  We generate an integral for every node associated with an unknown.
#
      for i in range ( 0, 2 ):

        test = element_node[i,element]

        bi, dbidx = basis_function ( test, element, node_x, element_q[quad] )
#
#  Consider the BASIS-th basis function, which is used to form the
#  value of the solution function.
#
        for j in range ( 0, 2 ):

          basis = element_node[j,element]

          bj, dbjdx = basis_function ( basis, element, node_x, element_q[quad] )

          c[test,basis] = c[test,basis] + element_w[quad] * bi * bj

  return c
def fem1d_heat_explicit_cfl ( x_num, k, x, dt ):

  cfl = 0.0
  for i in range ( 0, x_num - 2 ):
    cfl = max ( cfl, k[i+1] / ( x[i+1] - x[i] ) ** 2 )
 
  cfl = dt * cfl

  if ( 0.5 <= cfl ):
    print ( '' )
    print ( 'FEM1D_HEAT_EXPLICIT_CFL - Fatal error!' )
    print ( '  CFL condition failed.' )
    print ( '  0.5 <= K * dT / dX / dX = %g' % ( cfl ) )
    exit ( 'FEM1D_HEAT_EXPLICIT_CFL - Fatal error!' )

  return cfl 
def assemble_fem ( x_num, x, element_num, element_node, quad_num, t, k_fun, \
  rhs_fun ):
#
#  Initialize the arrays.
#
  b = np.zeros ( x_num )
  a = np.zeros ( ( x_num, x_num ) )
#
#  Get the quadrature weights and nodes.
#
  reference_w, reference_q = quadrature_set ( quad_num )
#
#  Consider each ELEMENT.
#
  for element in range ( 0, element_num ):

    element_x = np.zeros ( 2 )
    element_x[0] = x[element_node[0,element]]
    element_x[1] = x[element_node[1,element]]

    element_q = reference_to_physical ( element, element_node, x, quad_num, \
      reference_q )

    element_area = element_x[1] - element_x[0]

    element_w = np.zeros ( quad_num )
    for quad in range ( 0, quad_num ):
      element_w[quad] = ( element_area / 2.0 ) * reference_w[quad]
#
#  Consider the QUAD-th quadrature point in the element.
#
    k_value = k_fun ( quad_num, element_q, t )
    rhs_value = rhs_fun ( quad_num, element_q, t )

    for quad in range ( 0, quad_num ):
#
#  Consider the TEST-th test function.
#
#  We generate an integral for every node associated with an unknown.
#
      for i in range ( 0, 2 ):

        test = element_node[i,element]

        bi, dbidx = basis_function ( test, element, x, element_q[quad] )

        b[test] = b[test] + element_w[quad] * rhs_value[quad] * bi
#
#  Consider the BASIS-th basis function, which is used to form the
#  value of the solution function.
#
        for j in range ( 0, 2 ):

          basis = element_node[j,element]

          bj, dbjdx = basis_function ( basis, element, x, element_q[quad] )

          a[test,basis] = a[test,basis] + element_w[quad] * ( \
            + k_value[quad] * dbidx * dbjdx )

  return a, b

def fem1d_heat_explicit ( x_num, x, t, dt, k_fun, rhs_fun, bc_fun, \
  element_num, element_node, quad_num, mass, u ):
#
#  Check stability condition.
#
  k_vec = k_fun ( x_num, x, t )
  cfl = fem1d_heat_explicit_cfl ( x_num, k_vec, x, dt )
#
#  Compute the spatial finite element information.
#
  a, b = assemble_fem ( x_num, x, element_num, element_node, \
    quad_num, t, k_fun, rhs_fun )
#
#  The system we want to solve is
#
#    MASS * dudt = - A * u + b
#
#  Add "-A*u" to the right hand side;
#
  rhs = - np.dot ( a, u )
  for i in range ( 0, x_num ):
    rhs[i] = rhs[i] + b[i]
#
#  Now solve MASS * dudt = - A * u + b
#
  dudt = np.linalg.solve ( mass, rhs )
#
#  Set u_new = u + dt * dudt.
#
  u_new = np.zeros ( x_num )
  for i in range ( 0, x_num ):
    u_new[i] = u[i] + dt * dudt[i]
#
#  Impose boundary conditions on u_new.
#
  u_new = bc_fun ( x_num, x, t + dt, u_new )

  return u_new
def fem_solve(k,m,n,a,b,T):
  
  x_num = m+1
  x_min = a
  x_max = b
  L = x_max - x_min
  dx = L / ( x_num - 1 )
  x = np.linspace ( x_min, x_max, x_num )
#
#  Set the times.
#
  t_num = n+1
  t_min = 0.0
  t_max = T
  dt = ( t_max - t_min ) / ( t_num - 1 )
  t = np.linspace ( t_min, t_max, t_num )
#
#  Set finite element information.
#
  element_num = x_num - 1
  element_node = np.zeros ( ( 2, element_num ) )
  for j in range ( 0, element_num ):
    element_node[0,j] = j
    element_node[1,j] = j + 1
  quad_num = 3
  mass = assemble_mass ( x_num, x, element_num, element_node, quad_num )

  print ( '' )
  print ( '  Number of X nodes = %d' % ( x_num ) )
  print ( '  X interval = [ %f, %f ]' % ( x_min, x_max ) )
  print ( '  X step size = %f' % ( dx ) )
  print ( '  Number of T steps = %d' % ( t_num ) )
  print ( '  T interval = [ %f, %f ]' % ( t_min, t_max ) )
  print ( '  T step size = %f' % ( dt ) )
  print ( '  Number of elements = %d' % ( element_num ) )
  print ( '  Number of quadrature points = %d' % ( quad_num ) )
  #test01 standard normal initial
  
  #u = np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
  #bc_test = bc_test01
  
  #test02 uniform initial
  c = 1.0
  u = c*np.ones(x_num)
  bc_test = bc_test02
  
  ######Solving and plotting
  #plt.plot(x,u,'b',label="Initial")
  plt.ylim(0,max(u)+0.2)
  #test 2 true
  for j in range ( 1, t_num ):

      u = fem1d_heat_explicit ( x_num, x, t[j-1], dt, k_test01, \
        rhs_test01, bc_test, element_num, element_node, quad_num, mass, u )
  #test 1 true
  #u_true = np.exp(-x**2 / (2+4*k*t_max)) / np.sqrt((2+4*k*t_max)*np.pi)  
  
  u_true = (4*c/np.pi)*np.sin(np.pi*(x-x_min)/L)*np.exp(-k*t_max*np.pi**2/L**2)
  for i in range(1,17):
      n = 2*i+1
      u_true+=(4*c/(n*np.pi))*np.sin(n*np.pi*(x-x_min)/L)*np.exp(-k*t_max*n**2*np.pi**2/L**2)
     
  
  print "Error at t="+str(t_max)+":"+str(np.linalg.norm((u-u_true)*dx,ord=1))
   
  plt.plot(x,u,'ro',label="FEM, dx="+str(dx))
  plt.plot(x,u_true,'k',label="true")
  plt.legend(loc='lower center')
  
  plt.show()
  return np.linalg.norm((u-u_true)*dx,ord=1)

#order of accuracy study
def error_test02():
    k = 1.0
    T = 0.1
    a = 0.0
    b = 10.0
    dx = []
    e = []
    for m in [100,200,300]:
        dx.append((b-a)/m)
        n = m*m/100
        error = fem_solve(k,m,n,a,b,T)
        e.append(error) 
#test 1
#fem_solve(1.0,100,400,-5.0,5.0,1.0)
#test 2
#fem_solve(1.0,100,100,0.0,10.0,0.1)
#error_test02()