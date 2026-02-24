#include<iostream>
#include<ctime>
int main()
{
    freopen("input.txt","w",stdout);
    srand(42);  // 固定种子，使随机数可复现
    for(int i=0; i< 10000; ++i)
    {
        int a = rand() % 1000 + 1;
        int b = rand() % 1000 + 1;
        
        std::cout << a << "+" << b << "=" << (a+b) << std::endl;
    }
    return 0;
}
