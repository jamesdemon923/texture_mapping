
#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        u = std::fmin(1, std::fmax(u, 0));
        v = std::fmin(1, std::fmax(v, 0));
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        float u00_u = int(u * width), u00_v = int(v * height);
        float u01_u = u00_u, u01_v = u00_v + 1;
        float u10_u = u00_u + 1, u10_v = u00_v;
        float u11_u = u00_u + 1, u11_v = u00_v + 1;

        Eigen::Vector3f color00, color01, color10, color11, color1, color0, color;
        color00 = getColor(u00_u / width, u00_v / height);
        color10 = getColor(u01_u / width, u01_v / height);
        color01 = getColor(u10_u / width, u10_v / height);
        color11 = getColor(u11_u / width, u11_v / height);
        color0 = color00 + (color10 - color00) * (u * width - u00_u);
        color1 = color01 + (color11 - color01) * (u * width - u01_u);
        color = color0 + (color1 - color0) * (v * height - u00_v);
        return color;
    }

};
#endif //RASTERIZER_TEXTURE_H
