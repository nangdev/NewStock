import { api } from 'api/axiosInstance';
import { API_PATH } from 'constants/api';
import {
  NotificationDeleteResType,
  NotificationListResType,
  NotificationReadResType,
} from 'types/api/notification';

export const getNotificationList = async () => {
  const response = await api.get<NotificationListResType>(API_PATH.NOTIFICATION.ALL);
  return response.data;
};

export const putNotificationRead = async ({ unId }: { unId: number }) => {
  const response = await api.put<NotificationReadResType>(`${API_PATH.NOTIFICATION.READ}/${unId}`);
  return response.data;
};

export const deleteNotification = async ({ unId }: { unId: number }) => {
  const response = await api.delete<NotificationDeleteResType>(
    `${API_PATH.NOTIFICATION.DELETE}/${unId}`
  );
  return response.data;
};
