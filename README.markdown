# Render the cow with different mapping

## 1. Setup

**Operating & compiling environment**:

* Ubuntu (WSL2 in Windows11)

**Library**:

* Eigen
* OpenCV

## 2. Implement the rasterization function

The basic idea is to project the vertices of a triangle into 2D space, and then color the pixels within that triangle appropriately by using the interpolation to calculate various attributes like color, normal vectors, texture coordinates.

```c++
Void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
    // About the correction about depth, color, normal, texcoords:
    // Here we use toVector4, w is directly equal to 1
    // and the correction formula degenerates directly into a direct interpolation of the depth of the triangle vertices in NDC space using alpha beta gamma
    auto v = t.toVector4();

    // Find out the bounding box of current triangle.
    int boundingbox_x_left = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    int boundingbox_x_right = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    int boundingbox_y_left = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    int boundingbox_y_right = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (auto x = boundingbox_x_left; x <= boundingbox_x_right; x++) {
        for (auto y = boundingbox_y_left; y <= boundingbox_y_right; y++) {
            if (insideTriangle((float)x+0.5, (float)y+0.5, t.v)) {
                // Inside your rasterization loop:
                // * v[i].w() is the vertex view space depth value z.
                // * Z is interpolated view space depth for the current pixel
                // * zp is depth between zNear and zFar, used for z-buffer
                auto[alpha, beta, gamma] = computeBarycentric2D((float)x+0.5, (float)y+0.5, t.v);
                float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                zp *= Z;
                
                // set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                if (zp < depth_buf[get_index(x, y)]) {

                    // Interpolate the attributes
                    auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                    auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                    auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                    // What is shadingcoords, please refer to http://games-cn.org/forums/topic/zuoye3-interpolated_shadingcoords/
                    auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);

                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;
                    // Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
                    auto pixel_color = fragment_shader(payload);

                    depth_buf[get_index(x, y)] = zp;
                    set_pixel(Eigen::Vector2i(x, y), pixel_color);
                }
            }
        }
    }
}
```

## 3. Rendering the cow with Blinn-Phong reflect model

The Blinn-Phong reflect model: 



$$
L = L_{a}+L_{d}+L_{s}=k_{a}I_{a}+k_{d}(I/r^{2})max(0,\vec{n} \cdot \vec{l})+k_{s}(I/r^{2})max(0,\vec{n} \cdot \vec{h})^{p}
$$



So, we first calculate the $\vec{l}$, $\vec{n}$, $\vec{v}$, $\vec{h}$:

```c++
Eigen::Vector3f light_vector = (light.position - point).normalized();
Eigen::Vector3f view_vector = (eye_pos - point).normalized();
Eigen::Vector3f half_vector = (light_vector + view_vector).normalized();
Eigen::Vector3f n = normal.normalized();
```

Then, we get the $L_{a}$, $L_{d}$, $L_{s}$:

```c++
// The ambient light
Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity); // cwiseProduct() is a component-wise multiplication operation provided by the Eigen library in C++

// The diffuse light
Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, n.dot(light_vector));

// The specular light
Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / r2) * std::pow(std::max(0.0f, n.dot(half_vector)), p);
```

## 4. Rendering the cow with texture mapping

Replace kd with texture_color, and we should use UV coordinates when getting color of textures:

```c++
if (payload.texture)
{
    return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
}
Eigen::Vector3f texture_color;
texture_color << return_color.x(), return_color.y(), return_color.z();
```

where **payload** is a struct which contains information such as texture, and it is defined in the shader.hpp

```c++
struct fragment_shader_payload
{
    fragment_shader_payload()
    {
        texture = nullptr;
    }

    fragment_shader_payload(const Eigen::Vector3f& col, const Eigen::Vector3f& nor,const Eigen::Vector2f& tc, Texture* tex) :
         color(col), normal(nor), tex_coords(tc), texture(tex) {}


    Eigen::Vector3f view_pos;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;
    Eigen::Vector2f tex_coords;
    Texture* texture;
};
```

payload.texture is to take the texture of the struct payload

## 5. Rendering the cow with bump mapping

The formula in the sample code:


$$
dU = kh * kn * (h(u+1/w,v) - h(u,v))
$$

$$
dV = kh * kn * (h(u,v+1/h)-h(u,v))
$$



Where $kh * kn$ is the influence factor (constant), which indicates the influence of texture normals on the object, that is, c1c2 in the figure below.

<div align=center>
    <img src="images\normal mapping.png" width="500"/>
</div>




h(), in the normal(bump) mapping, is the color of the coordinate (u,v) corresponding to the vertex (RGB value)

Why do we use "u+1.0f/w" instead of directly "u+1" can be explained by a part of the definition of getColor() in Texture.hpp:

```c++
auto u_img = u * width.
auto v_img = (1 - v) * height.
auto color = image_data.at<cv::Vec3b>(v_img, u_img).
return Eigen::Vector3f(color[0], color[1], color[2]).
```

The u,v values here all multiply the width and height of the texture, and if you transform them, moving "u*width+1" instead of one unit, so 1 unit in our function should correspond to 1/width, 1/h is the same.

So my code is:

```c++
float u = payload.tex_coords.x();
float v = payload.tex_coords.y();
float w = payload.texture->width;
float h = payload.texture->height;
float dU = kh * kn * (payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
float dV = kh * kn * (payload.texture->getColor(u, v + 1 / h).norm() - payload.texture->getColor(u, v).norm());
```

.norm() is a function defined in the Eigen library to find the parametric number, which is the sum of squares of all elements and then squared. The vector's norm is the size of the original set. getColor returns a vector that stores color values: (color[0], color[1], color[2]), which correspond to RGB values, while $dU$ and $dV$ are a float value, not a Vector. To achieve the real height value represented by h(), we need to use norm.() to map the vector to a real number.

## 6. Rendering the cow with displacement mapping

<div align=center>
    <img src="images\displacement mapping.png" width="500"/>
</div>



So, I add the step of moving point:

```c++
point += kn * normal * payload.texture->getColor(u, v).norm();
```

## 7. Bilinear interpolation

I use the bilinear interpolation to make the texture transition smoother.

<div align=center>
    <img src="images\bilinear interpolation.png" width="500"/>
</div>



In C++, **int** rounding is straightforward by rounding off the decimal part, so int(u * width) and int(v * height) get the point in the bottom left corner.

```c++
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
```

The default texture map coordinate range is [0, 1]^2, we need to restrict the u,v coordinate range in getColor()

```c++
u = std::fmin(1, std::fmax(u, 0));
v = std::fmin(1, std::fmax(v, 0));
```

## 8. The result

### 1. ./Rasterizer output.png normal

<div align=center>
    <img src="images\result1.png" width="500"/>
</div>


### 2. ./Rasterizer output.png phong

<div align=center>
    <img src="images\result2.png" width="500"/>
</div>


### 3. ./Rasterizer output.png texture

<div align=center>
    <img src="images\texture mapping.png" width="500"/>
</div>


### 4. ./Rasterizer output.png bump

<div align=center>
    <img src="images\result_bump.png" width="500"/>
</div>


### 5. ./Rasterizer output.png displacement

<div align=center>
    <img src="images\result_displacement.png" width="500"/>
</div>


### 6. With bilinear interpolation

| Before bilinear interpolation                          | After bilinear interpolation                         |
| ------------------------------------------------------ | ---------------------------------------------------- |
| ![Before bilinear interpolation](images/before_bi.png) | ![After bilinear interpolation](images/after_bi.png) |

### 7. Other models

<div align=center>
    <img src="images\other model.png" width="500"/>
</div>

