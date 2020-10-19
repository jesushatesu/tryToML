//#include <iostream>
//#include <cmath>
//#include <cstdlib>
//#include <fstream>
//
//
//int main()
//{
//	std::ofstream out;
//	out.open("DataTest.txt");
//	
//	out << "topology: 2 3 1" << std::endl;
//	for (int i = 2000; i >= 0; i--)
//	{
//		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
//		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
//		int t = !(n1 & n2);
//
//		out << "in: " << n1 << ".0 " << n2 << ".0" << std::endl;
//		out << "out: " << t << ".0 " << std::endl;
//	}
//
//	out.close();
//}