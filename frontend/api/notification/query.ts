import { useMutation, useQuery } from '@tanstack/react-query';

import { deleteNotification, getNotificationList, putNotificationRead } from '.';

export const useNotificationListQuery = () => {
  return useQuery({
    queryKey: ['notificationList'],
    queryFn: getNotificationList,
  });
};

export const useNotificationReadMutation = () => {
  return useMutation({
    mutationFn: putNotificationRead,
    onSuccess: () => {
      console.log('알림 읽기 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useNotificationDeleteMutation = () => {
  return useMutation({
    mutationFn: deleteNotification,
    onSuccess: () => {
      console.log('알림 삭제 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};
