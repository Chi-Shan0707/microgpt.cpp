#include <iostream>
#include <locale>
#include <string>
#include <set>
#include <cwctype> 

using namespace std;

int main()
{
    // 设置全局 locale 为 UTF-8 以支持宽字符 I/O
    locale::global(locale("C.utf8"));
    // ** 这个因不同平台不同电脑而异，可能需要调整为 "en_US.UTF-8" 或其他适合的 UTF-8 locale 名称 **

    
    wcin.imbue(locale());
    wcout.imbue(locale());

    // 使用 set<wchar_t> 存储唯一的 Unicode 码点（字符）
    set<wchar_t> u_chars;
    wstring line;
    while (getline(wcin, line)) 
    {
        for (auto const & ch : line) 
        {
            u_chars.insert(ch);
        }
    }

    // 示例：打印出收集到的字符数量和这些字符
    wcout << L"Unique characters collected: " << u_chars.size() << endl;
    for (auto const & ch : u_chars) {
        wcout << ch;
    }
    wcout << endl;

    return 0;
}
