#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

void dfsFolder(const std::string& folder_path,
               std::vector<std::string>& input_image_list) {
  DIR* pdir;
  struct dirent* ptr;
  if (!(pdir = opendir(folder_path.c_str()))) {
    std::cout << "can not match the folder path: " << folder_path << std::endl;
    exit(-1);
  }

  while ((ptr = readdir(pdir)) != 0) {
    struct stat st;
    if (fstatat(dirfd(pdir), ptr->d_name, &st, 0) < 0) {
      std::cout << "failed to open access file or diectory: "
                << std::string(ptr->d_name) << std::endl;
      exit(-1);
    }
    if (S_ISDIR(st.st_mode)) {
      if ((strcmp(ptr->d_name, ".") != 0) && (strcmp(ptr->d_name, "..") != 0)) {
        std::string newPath = folder_path + "/" + ptr->d_name;
        dfsFolder(newPath, input_image_list);
      }
    } else {
      input_image_list.push_back(ptr->d_name);
    }
  }
  closedir(pdir);
}