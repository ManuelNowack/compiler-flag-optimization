//===============================
// Easy printing functionalities
//===============================

#ifndef DEBUGGING_H
#define DEBUGGING_H

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>

#define trace(x) std::cout << (x) << std::endl;

template <typename T>
std::ostream &operator<<(std::ostream &output, const std::vector<T> &x);
template <typename key_t, typename T>
std::ostream &operator<<(std::ostream &output, const std::map<key_t, T> &x);
template <typename key_t, typename T>
std::ostream &operator<<(std::ostream &output, const std::unordered_map<key_t, T> &x);

template <typename T, typename V>
std::ostream &operator<<(std::ostream &output, const std::pair<T, V> &x) {
    output << '(' << x.first << ", " << x.second << ')';
    return output;
}

template <typename T>
std::ostream &operator<<(std::ostream &output, const std::vector<T> &x) {
    if (x.empty()) {
        output << "[]";
        return output;
    }
    auto iter = x.begin();
    output << '[' << *iter;
    iter++;
    for (; iter != x.end(); iter++) {
        output << ", " << *iter;
    }
    output << ']';
    return output;
}

template <typename key_t, typename T>
std::ostream &operator<<(std::ostream &output, const std::map<key_t, T> &x) {
    if (x.empty()) {
        output << "{}";
        return output;
    }
    auto iter = x.begin();
    output << '{' << iter->first << ": " << iter->second;
    iter++;
    for (; iter != x.end(); iter++) {
        output << ", " << iter->first << ": " << iter->second;
    }
    output << '}';
    return output;
}

template <typename key_t, typename T>
std::ostream &operator<<(std::ostream &output, const std::unordered_map<key_t, T> &x) {
    if (x.empty()) {
        output << "{}";
        return output;
    }
    auto iter = x.begin();
    output << '{' << iter->first << ": " << iter->second;
    iter++;
    for (; iter != x.end(); iter++) {
        output << ", " << iter->first << ": " << iter->second;
    }
    output << '}';
    return output;
}

#endif
