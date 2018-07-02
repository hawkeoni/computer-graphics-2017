//Where else do i need to switch z/y?
//Why my cosine is bad
//Fill the rest and make materials

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdlib.h>

#include "rtcore_ray.h"
#include "rtcore.h"
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtx/rotate_vector.hpp"
#include "OBJ_Loader.h"
#include "EasyBMP.h"

#define N 100
#define ALPHA 6

using namespace std;



BMP gray_world(BMP srcImage) {
    uint S, Sr = 0, Sg = 0, Sb = 0;
    uint height = srcImage.TellHeight(), width = srcImage.TellWidth();
    uint i, j;
    uint new_red, new_gre, new_blu;
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
          RGBApixel p = srcImage.GetPixel(j, i);
            Sr += p.Red;
            Sg += p.Green;
            Sb += p.Blue;
        }
    }
    Sr /= height * width;
    Sg /= height * width;
    Sb /= height * width;
    S = (Sr + Sg + Sb) / 3;
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
          RGBApixel p = srcImage.GetPixel(j, i);
          new_red = p.Red * S / Sr;
          new_red = (new_red < 255) ? new_red : 255;
          new_gre = p.Green * S / Sg;
          new_gre = (new_gre < 255) ? new_gre : 255;
          new_blu = p.Blue * S / Sb;
          new_blu = (new_blu < 255) ? new_blu : 255;
          p.Red = new_red;
          p.Green = new_gre;
          p.Blue = new_blu;
          srcImage.SetPixel(j, i, p);
        }
    }
    return srcImage;
}


float vgrid[N][N][N];
float minpower = 0.00000001;
glm::vec3 c_min, c_max;
glm::vec3 v;
int cntr = 0;

short *materials;

void Boxfilter(float vgrid[N][N][N]){
  float ngrid[N][N][N];
  for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++)
          ngrid[i][j][k] = (

            vgrid[i - 1][j - 1][k - 1] + vgrid[i - 1][j - 1][k] + vgrid[i - 1][j - 1][k + 1] + \
            vgrid[i - 1][j][k - 1] + vgrid[i - 1][j][k] + vgrid[i - 1][j][k + 1] + \
            vgrid[i - 1][j + 1][k - 1] + vgrid[i - 1][j + 1][k] + vgrid[i - 1][j + 1][k + 1] + \

            vgrid[i][j - 1][k - 1] + vgrid[i][j - 1][k] + vgrid[i][j - 1][k + 1] + \
            vgrid[i][j][k - 1] + vgrid[i][j][k] + vgrid[i][j][k + 1] + \
            vgrid[i][j + 1][k - 1] + vgrid[i][j + 1][k] + vgrid[i][j + 1][k + 1] + \

            vgrid[i + 1][j - 1][k - 1] + vgrid[i + 1][j - 1][k] + vgrid[i + 1][j - 1][k + 1] + \
            vgrid[i + 1][j][k - 1] + vgrid[i + 1][j][k] + vgrid[i][j][k + 1] + \
            vgrid[i + 1][j + 1][k - 1] + vgrid[i + 1][j + 1][k] + vgrid[i + 1][j + 1][k + 1]\
            ) / 9;
  for (int i = 1; i < N - 1; i++)
    for (int j = 1; j < N - 1; j++)
      for (int k = 1; k < N - 1; k++){
        vgrid[i][j][k] = ngrid[i][j][k];
      }
}


void printp(RGBApixel p){
  cout << short(p.Red) << " " << short(p.Green) << " " << short(p.Blue) << endl;
}


float powtodb(float power){
  return log(power) / log(10) * 10;
}

RGBApixel alphablending(RGBApixel a, RGBApixel b){
  float alpha = ALPHA /255.f;
  a.Red = alpha * b.Red + (1 - alpha) * a.Red;
  a.Green = alpha * b.Green+ (1 - alpha) * a.Green;
  a.Blue = alpha * b.Blue + (1 - alpha) * a.Blue;
  return a;
}

RGBApixel slave(RGBApixel a, RGBApixel b, float t){
  RGBApixel res;
  res.Red = a.Red + (b.Red - a.Red) * t;
  res.Green = a.Green + (b.Green - a.Green) * t;
  res.Blue = a.Blue + (b.Blue - a.Blue) * t;
  res.Alpha = ALPHA;
  return res;
}

RGBApixel color_interpolate(float p){
  float db = powtodb(p);
  //cout << db << endl;
  //cout << db << endl;
  RGBApixel a, b;
  a.Alpha = ALPHA; b.Alpha = ALPHA;
  if (db <= -80){
    a.Red = 0;
    a.Green = 0;
    a.Blue = 0;
    return a;
  }
  if (db <= -70){
    a.Red = 0;
    a.Green = 0;
    a.Blue = 0;
    b.Red = 51;//51, 51, 204)
    b.Green = 51;
    b.Blue = 204;
    return slave(a, b, (db + 80.f) / 10.f);
  }
  if (db <= -60){
    a.Red = 102;
    a.Green = 0;
    a.Blue = 102;
    b.Red = 204;
    b.Green = 0;
    b.Blue = 153;
    return slave(a, b, (db + 70.f) / 10.f); 
  }
  if (db <= -50){
    a.Red = 204;
    a.Green = 0;
    a.Blue = 153; //255, 51, 153
    b.Red = 255;
    b.Green = 51;
    b.Blue = 153;
    return slave(a, b, (db + 60.f) / 10.f); 
  }
  if (db <= -40){
    a.Red = 255;
    a.Green = 51;
    a.Blue = 153;
    b.Red = 255;
    b.Green = 0;
    b.Blue = 102;
    //cout << 5 << endl;
    return slave(a, b, (db + 50.f) / 10.f);
  }
  if (db <= -30){
    a.Red = 255;
    a.Green = 0;
    a.Blue = 102;
    b.Red = 255;
    b.Green = 80;
    b.Blue = 80;
    //cout << 6 << endl;
    return slave(a, b, (db + 40.f)/ 10.f);
  }
  if (db <= -20){ //Yellow to white
    a.Red = 255;
    a.Green = 80;
    a.Blue = 80;
    b.Red = 255;
    b.Green = 102;
    b.Blue = 0;
    //cout << 7 << endl;
    return slave(a, b, (db + 30.f) / 10.f); 
  }
  if (db <= -10){ //Yellow to white
    a.Red = 255;
    a.Green = 102;
    a.Blue = 0;
    b.Red = 255; //255, 153, 51)
    b.Green = 153;
    b.Blue = 51;
    //cout << 7 << endl;
    return slave(a, b, (db + 20.f) / 10.f); 
  }
  if (db <= 0){ //Yellow to white
    a.Red = 255;
    a.Green = 153; //gb(255, 204, 102)
    a.Blue = 51;
    b.Red = 255;
    b.Green = 204;
    b.Blue = 102;
    //cout << 7 << endl;
    return slave(a, b, (db + 10.f) / 10.f); 
  }
  if (db <= 10){ //Yellow to white
    a.Red = 255; ////(255, 255, 102)
    a.Green = 204;
    a.Blue = 102;
    b.Red = 255;
    b.Green = 255;
    b.Blue = 102;
    //cout << 7 << endl;
    return slave(a, b, db / 10.f); 
  }
  if (db <= 20){ //Yellow to white
    a.Red = 255;
    a.Green = 255;
    a.Blue = 102;
    b.Red = 255;
    b.Green = 255;
    b.Blue = 255;
    //cout << 7 << endl;
    return slave(a, b, (db - 10.f) / 10.f); 
  }

  return a;
}


void printv3(glm::vec3 v){
  cout << "{" << v.x << ", " << v.y << ", " << v.z << "}" << endl;
}

void printv3i(glm::vec3 v){
  cout << "{" << int(v.x) << ", " << int(v.y) << ", " << int(v.z) << "}" << endl;
}

float vec_cos(glm::vec3 a, glm::vec3 b){
  return glm::dot(a, b) / (glm::length(a) * glm::length(b));
}

void tofloat(float * vec, glm::vec3 g){
  vec[0] = g.x;
  vec[1] = g.y;
  vec[2] = g.z;
}

glm::vec3 toglm(float *vec){
  glm::vec3 res;
  res.x = vec[0];
  res.y = vec[1];
  res.z = vec[2];
  return res;
}

struct Vertex{
	float x, y, z, a;
};

struct Triangle{
	int v0, v1, v2;
};

unsigned int addMesh(RTCScene scene_i, objl::Mesh msh){
  if (msh.MeshMaterial.name == "" || msh.MeshMaterial.name == "None") {
    materials[cntr++] = 0;
  }
  else materials[cntr++] = 1;
  unsigned int mesh = rtcNewTriangleMesh (scene_i, RTC_GEOMETRY_STATIC, msh.Indices.size() / 3, msh.Vertices.size());
  Vertex* vertices = (Vertex*)rtcMapBuffer(scene_i,mesh,RTC_VERTEX_BUFFER);
  for (int i = 0; i < msh.Vertices.size(); i++){
    vertices[i].x = msh.Vertices[i].Position.X;
    vertices[i].y = msh.Vertices[i].Position.Z;
    vertices[i].z = msh.Vertices[i].Position.Y;
  }
  rtcUnmapBuffer(scene_i, mesh, RTC_VERTEX_BUFFER);
  Triangle* triangles = (Triangle*)rtcMapBuffer(scene_i, mesh, RTC_INDEX_BUFFER);
  for (int i = 0; i < msh.Indices.size(); i += 3){
    triangles[i / 3].v0 = msh.Indices[i];
    triangles[i / 3].v1 = msh.Indices[i + 1];
    triangles[i / 3].v2 = msh.Indices[i + 2];
  }
  rtcUnmapBuffer(scene_i,mesh,RTC_INDEX_BUFFER);
  return mesh;

}

void rayvfill(RTCScene scene, glm::vec3 src_pos, glm::vec3 dir, float maxpower_arg, float distance){

  RTCRay vray;
  if (maxpower_arg < minpower) return;
  float maxpower = maxpower_arg;
  tofloat(vray.org, src_pos);
  tofloat(vray.dir, dir);
  vray.tnear = 0.0f;
  vray.tfar = std::numeric_limits<decltype(vray.tfar)>::infinity();
  vray.geomID = RTC_INVALID_GEOMETRY_ID;
  vray.primID = RTC_INVALID_GEOMETRY_ID;
  vray.mask = -1;
  vray.time = 0.0f;
  rtcIntersect(scene, vray);
  if (vray.geomID != RTC_INVALID_GEOMETRY_ID){
    glm::vec3 hit = src_pos + dir * vray.tfar;
    float m = 0;
    //printv3(hit);
    while (m < vray.tfar){
      glm::vec3 curpos = src_pos + dir * m, curmesh;
      curmesh = (curpos - c_min) / v; //is the problem here?
      if (glm::length(curpos - src_pos) > max(glm::length(v), 1.f)){
        maxpower = maxpower_arg / max(powf(glm::length(curpos - src_pos), 4) , 1.0f);
      }
      if (maxpower <= minpower) {
        break;
      }
      if (maxpower > vgrid[int(curmesh.x)][int(curmesh.y)][int(curmesh.z)]){
        vgrid[int(curmesh.x)][int(curmesh.y)][int(curmesh.z)] = maxpower;
      }
      m += distance;
    }
    if (m >= vray.tfar && maxpower > minpower){
      //No material = solid wall, we reflect
      //Material = glass, we go through
      if (materials[vray.geomID] == 0){
        glm::vec3 refldir;
        refldir = glm::reflect(dir, toglm(vray.Ng));
        refldir = glm::normalize(refldir);
        //cout << "reflect " << endl;
        rayvfill(scene, hit - dir * distance * 5.f, refldir, maxpower, distance);
        //cout << "out of it"<< endl;
      }
      else {
        rayvfill(scene, hit + dir * distance * 5.f , dir, maxpower - 4, distance);
      }
    }
  }
  else{
    float m = 0;
    //we should go untill out of image
    while ((src_pos.x + m * dir.x < c_max.x) and (src_pos.x + m * dir.x > c_min.x) and (src_pos.y + m * dir.y < c_max.y) \
    and (src_pos.y + m * dir.y > c_min.y) and (src_pos.z + m * dir.z < c_max.z) and \
    (src_pos.z + m * dir.z > c_min.z))
      {
        glm::vec3 curmesh, curpos = src_pos + dir * m;
        if (curmesh.x < 0 || curmesh.x >= 100 || curmesh.y < 0 || curmesh.y >= 100 || curmesh.z < 0 || curmesh.z >= 100) break;
        curmesh = (curpos - c_min) / v;
        if (glm::length(curpos - src_pos) > max(glm::length(v), 1.f)){
          maxpower = maxpower_arg / max(powf(glm::length(curpos - src_pos), 4) , 1.0f);
        }
        if (maxpower <= minpower) break;
        if (maxpower > vgrid[int(curmesh.x)][int(curmesh.y)][int(curmesh.z)]){
          vgrid[int(curmesh.x)][int(curmesh.y)][int(curmesh.z)] = maxpower;
        }
        m += distance;
      }
      return;
  }
  return;
}

int main(int argc, char **argv){
  if (argc < 2) {
    cout << "Please enter settings file" << endl;
    return 1;
  }
  std::ifstream infile(argv[1]);
  string temp, filename, savename;
  glm::vec3 cam_pos, src_pos;
  int W = 800, H = 600;
  infile >> temp >> W >> H;
  infile >> temp >> filename;
  infile >> temp >> cam_pos.x >> cam_pos.y >> cam_pos.z;
  infile >> temp >> savename;
  infile >> temp >> src_pos.x >> src_pos.y >> src_pos.z;
  RTCDevice g_device = rtcNewDevice(NULL);
  RTCScene g_scene = rtcDeviceNewScene(g_device, RTC_SCENE_STATIC,RTC_INTERSECT1);
  objl::Loader Loader;
  bool loadout;
  loadout = Loader.LoadFile("../assets/" + filename);
  materials = (short*)realloc(materials, sizeof(short) * Loader.LoadedMeshes.size());
  for (int i = 0; i < Loader.LoadedMeshes.size(); i++){
    objl::Mesh curMesh = Loader.LoadedMeshes[i];
      for (int j = 0; j < curMesh.Vertices.size(); j++){
        float x, y, z;
        x = curMesh.Vertices[j].Position.X;
        y = curMesh.Vertices[j].Position.Y;
        z = curMesh.Vertices[j].Position.Z;
        c_min.x = c_min.x < x ? c_min.x : x;
        c_min.y = c_min.y < z ? c_min.y : z;
        c_min.z = c_min.z < y ? c_min.z : y;
        c_max.x = c_max.x > x ? c_max.x : x;
        c_max.y = c_max.y > z ? c_max.y : z;
        c_max.z = c_max.z > y ? c_max.z : y;
      }
    }
  cout << "Size is: " << c_min.x << "x" << c_max.x << " " << c_min.y << "x" << c_max.y << " " << c_min.z << "x" << c_max.z << endl;
  for (int i = 0; i < Loader.LoadedMeshes.size(); i++){
    addMesh(g_scene, Loader.LoadedMeshes[i]);
  }
  rtcCommit(g_scene);

  //initialized vgrid
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
        vgrid[i][j][k] = minpower * 10;
      }
    }
  }
  v.x = (c_max.x - c_min.x) / N;
  v.y = (c_max.y - c_min.y) / N;
  v.z = (c_max.z - c_min.z) / N;
  //ray casting
  float distance = min(min(v.x, v.y), v.z);
  RTCRay vray;
  vray.org[0] = src_pos.x;
  vray.org[1] = src_pos.y;
  vray.org[2] = src_pos.z;
  for (float x = -1; x < 1; x += 1.0f / N){
    for (float y = -1; y < 1; y += 1.0f / N){
      for (float z = -1; z < 1; z += 1.0f / N){
        glm::vec3 dir(x, y, z);
        float l = glm::length(dir);
        if (l > 0.001) {
          rayvfill(g_scene, src_pos, glm::normalize(dir), 100, distance);
        }
      }
    }
  } 
  //Boxfilter(vgrid);
 	RTCRay ray;
  ray.org[0] = 10;
  ray.org[1] = 0;
  ray.org[2] = 0;
  ray.dir[0] = 0;
  ray.dir[1] = -1;
  ray.dir[2] = 0;
  ray.tnear = 0.0f;
  ray.tfar = std::numeric_limits<decltype(ray.tfar)>::infinity();
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
  ray.primID = RTC_INVALID_GEOMETRY_ID;
  ray.mask = -1;
  ray.time = 0.0f;

  /* intersect ray with scene */
  BMP image;
  
  image.SetSize(W, H);
  RGBApixel p;
  p.Red = 0;
  p.Green = 0;
  p.Blue = 0;
  unsigned short color = 0;

  for (int i = 0; i < H; i++)
      for (int j = 0; j < W; j++)
          image.SetPixel(j, i, p);

  p.Red = 255;
  p.Green = 255;
  p.Blue = 255;
  ray.org[0] = cam_pos.x;
  ray.org[1] = cam_pos.y;
  ray.org[2] = cam_pos.z;
  ray.tnear = 0.0f;
  ray.tfar = std::numeric_limits<decltype(ray.tfar)>::infinity();
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
  ray.primID = RTC_INVALID_GEOMETRY_ID;
  ray.mask = -1;
  ray.time = 0.0f;
  float cx, cy;
  glm::vec3 up(1, 0, 0), right(0, 1, 0), viewdir(0, 0, -1);
  float r1, r2, r3;
  infile >> temp >> r1 >> r2 >> r3;
  up *= W; right *= H; viewdir *= powf(W * W + H * H, 0.5) / 2;

  up = glm::rotate(up, r1, viewdir);
  right = glm::rotate(right, r1, viewdir);

  right = glm::rotate(right, r2, up);
  viewdir = glm::rotate(viewdir, r2, up);

  up = glm::rotate(up, r3, right);
  viewdir = glm::rotate(viewdir, r3, right);

  for (int i = 0; i < H; i++){
    cy = i;
    for (int j = 0; j < W; j++){
      cx = j;
      glm::vec3 cam = ((cx + 0.5f) / W - 0.5f) * up + ((cy + 0.5f) / H - 0.5f) * right + viewdir;
      tofloat(ray.dir, cam);
      ray.tnear = 0.0f;
      ray.tfar = std::numeric_limits<decltype(ray.tfar)>::infinity();
      ray.geomID = RTC_INVALID_GEOMETRY_ID;
      ray.primID = RTC_INVALID_GEOMETRY_ID;
      ray.mask = -1;
      ray.time = 0.0f;
      rtcIntersect(g_scene, ray);
      if (ray.geomID != RTC_INVALID_GEOMETRY_ID){
        glm::vec3 dir = toglm(ray.dir), norm = toglm(ray.Ng);
        float cosn = vec_cos(dir, norm);
        glm::vec3 pos = toglm(ray.org);
        glm::vec3 hit = pos + dir * ray.tfar;
        cosn = abs(cosn);
        p.Red = 255 * cosn;
        p.Green = 255 * cosn;
        p.Blue = 255 * cosn;
        float m = 0;
        dir = -dir;
        glm::vec3 where = hit + dir * m;
        dir = glm::normalize(dir);
        while (where.x < c_max.x && where.x > c_min.x && where.y < c_max.y && where.y > c_min.y && where.z < c_max.z && where.z > c_min.z){
          glm::vec3 curmesh = (where - c_min) / v;
          p = alphablending(p, color_interpolate(vgrid[int(curmesh.x)][int(curmesh.y)][int(curmesh.z)]));
          m += distance;
          where = hit + dir * m;
        }
        image.SetPixel(cx, cy, p);
      } 
    }
  }
  savename = "../img/" + savename;
  if (argc == 3 && string(argv[2]) == "--filter")
    gray_world(image).WriteToFile(savename.c_str());
  else
    image.WriteToFile(savename.c_str());
  rtcDeleteScene(g_scene);
	rtcDeleteDevice(g_device);
	return 0;
}



/*
  ray.geomID
  std::cout << "primID:                  " << ray.primID << std::endl;

*/