//
// Created by lenovo on 25-1-7.
//

#ifndef READDATA_H
#define READDATA_H

#include <vector>
#include <string>

namespace readData {
    class readData {
    public:
        static std::vector<std::vector<double> > readImageData(std::string path, int size = -1);

        static std::vector<int> readTagData(std::string path,int size = -1);
    };
}


#endif //READDATA_H
