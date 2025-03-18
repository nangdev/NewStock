FROM nginx:alpine

# Nginx 설정 파일 복사
COPY default.conf /etc/nginx/conf.d/default.conf

# 정적 파일이 필요하다면 추가할 수 있음 (선택 사항)
# COPY ./static /usr/share/nginx/html

# 80 포트 노출
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]