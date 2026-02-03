using System;
using System.IO;

internal static class Program
{
    public static int Main(string[] args)
    {
        File.WriteAllText("out.txt", "hello
");
        return 0;
    }
}
